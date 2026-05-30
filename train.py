import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import GPT2Config

import wandb
from model import EmotionInjectionTransformer


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_device_ids(device_cuda: str) -> List[int]:
    if device_cuda.lower() in ("", "none", "cpu"):
        return []
    return [int(device_id.strip()) for device_id in device_cuda.split(",") if device_id.strip()]


def setup_device(device_cuda: str) -> Tuple[torch.device, List[int]]:
    if not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU.")
        return torch.device("cpu"), []

    device_ids = parse_device_ids(device_cuda)
    if not device_ids:
        device_ids = list(range(torch.cuda.device_count()))

    available_count = torch.cuda.device_count()
    invalid_ids = [device_id for device_id in device_ids if device_id < 0 or device_id >= available_count]
    if invalid_ids:
        raise ValueError(
            f"Invalid CUDA device ids {invalid_ids}; this machine exposes {available_count} CUDA device(s)."
        )

    primary_device = torch.device(f"cuda:{device_ids[0]}")
    torch.cuda.set_device(primary_device)
    return primary_device, device_ids


def tensor_scalar(value) -> float:
    return float(torch.as_tensor(value).view(-1)[0].item())


def get_density(arousal_values: Iterable[float], valence_values: Iterable[float]) -> np.ndarray:
    va_values = np.stack([np.asarray(arousal_values), np.asarray(valence_values)], axis=1)
    kde = KernelDensity(kernel="gaussian", bandwidth="silverman")
    kde.fit(va_values)
    log_density = kde.score_samples(va_values)
    return np.exp(log_density)


class EmotionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "neutral_prompt_feature": item["neutral_prompt_feature"].to(torch.float32),
            "arousal": torch.as_tensor(item["arousal"], dtype=torch.float32).view(1),
            "valence": torch.as_tensor(item["valence"], dtype=torch.float32).view(1),
            "emotional_prompt_feature": item["emotional_prompt_feature"].to(torch.float32),
            "density": torch.as_tensor([item["density"]], dtype=torch.float32),
        }


@dataclass
class LossConfig:
    scale_factor: float
    enable_density: bool
    density_min: float
    density_max_weight: float
    enable_easa: bool
    easa_anchor_weight: float
    easa_smooth_weight: float
    easa_tau_negative: float
    easa_tau_positive: float


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def target_feature(batch: Dict[str, torch.Tensor], scale_factor: float) -> torch.Tensor:
    neutral_prompt_feature = batch["neutral_prompt_feature"]
    emotional_prompt_feature = batch["emotional_prompt_feature"]
    return (emotional_prompt_feature - neutral_prompt_feature) * scale_factor + neutral_prompt_feature


def emotion_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    density: torch.Tensor,
    config: LossConfig,
) -> torch.Tensor:
    if not config.enable_density:
        return F.mse_loss(prediction, target)

    density_weight = 1.0 / density.clamp_min(config.density_min).view(-1, 1, 1)
    if config.density_max_weight > 0:
        density_weight = density_weight.clamp_max(config.density_max_weight)
    return F.mse_loss(prediction * density_weight, target * density_weight)


def semantic_anchor_loss(
    neutral_feature: torch.Tensor,
    prediction: torch.Tensor,
    valence: torch.Tensor,
    tau_negative: float,
    tau_positive: float,
) -> torch.Tensor:
    neutral_flat = neutral_feature.flatten(start_dim=1)
    prediction_flat = prediction.flatten(start_dim=1)
    similarity = F.cosine_similarity(neutral_flat, prediction_flat, dim=1)

    negative_strength = (-valence.view(-1) / 3.0).clamp(min=0.0, max=1.0)
    tau = tau_positive + (tau_negative - tau_positive) * negative_strength
    return F.relu(tau - similarity).mean()


def lambda_smoothness_loss(
    easa_lambda: Optional[torch.Tensor],
    valence: torch.Tensor,
    arousal: torch.Tensor,
) -> torch.Tensor:
    if easa_lambda is None or easa_lambda.numel() < 2:
        reference = valence if easa_lambda is None else easa_lambda
        return reference.new_tensor(0.0)

    lambda_values = easa_lambda.view(-1)
    va_values = torch.cat([valence.view(-1, 1), arousal.view(-1, 1)], dim=1)
    lambda_diff = lambda_values - lambda_values.roll(shifts=1, dims=0)
    va_diff = va_values - va_values.roll(shifts=1, dims=0)
    va_distance = va_diff.pow(2).sum(dim=1).clamp_min(1e-6)
    return (lambda_diff.pow(2) / va_distance).mean()


def compute_losses(
    model,
    batch: Dict[str, torch.Tensor],
    config: LossConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    outputs = model(
        inputs_embeds=batch["neutral_prompt_feature"],
        arousal=batch["arousal"],
        valence=batch["valence"],
        return_easa_lambda=config.enable_easa,
    )
    prediction = outputs[0]
    easa_lambda = outputs[-1] if config.enable_easa else None
    target = target_feature(batch, config.scale_factor)

    loss_emo = emotion_loss(prediction, target, batch["density"], config)
    loss_anchor = prediction.new_tensor(0.0)
    loss_smooth = prediction.new_tensor(0.0)

    if config.enable_easa and config.easa_anchor_weight > 0:
        loss_anchor = semantic_anchor_loss(
            neutral_feature=batch["neutral_prompt_feature"],
            prediction=prediction,
            valence=batch["valence"],
            tau_negative=config.easa_tau_negative,
            tau_positive=config.easa_tau_positive,
        )
    if config.enable_easa and config.easa_smooth_weight > 0:
        loss_smooth = lambda_smoothness_loss(
            easa_lambda=easa_lambda,
            valence=batch["valence"],
            arousal=batch["arousal"],
        )

    total_loss = (
        loss_emo
        + config.easa_anchor_weight * loss_anchor
        + config.easa_smooth_weight * loss_smooth
    )

    lambda_mean = float("nan")
    lambda_min = float("nan")
    lambda_max = float("nan")
    if easa_lambda is not None:
        detached_lambda = easa_lambda.detach()
        lambda_mean = detached_lambda.mean().item()
        lambda_min = detached_lambda.min().item()
        lambda_max = detached_lambda.max().item()

    return total_loss, {
        "loss": total_loss.item(),
        "loss_emo": loss_emo.item(),
        "loss_anchor": loss_anchor.item(),
        "loss_smooth": loss_smooth.item(),
        "lambda_mean": lambda_mean,
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
    }


def average_metrics(metric_sums: Dict[str, float], total_count: int) -> Dict[str, float]:
    return {key: value / max(total_count, 1) for key, value in metric_sums.items()}


def accumulate_metrics(metric_sums: Dict[str, float], metrics: Dict[str, float], batch_size: int):
    for key, value in metrics.items():
        if np.isfinite(value):
            metric_sums[key] = metric_sums.get(key, 0.0) + value * batch_size


def train_one_epoch(model, train_loader, optimizer, device, config: LossConfig) -> Dict[str, float]:
    model.train()
    metric_sums: Dict[str, float] = {}
    total_count = 0

    for batch in tqdm(train_loader, desc="Training"):
        batch = move_batch_to_device(batch, device)
        batch_size = batch["neutral_prompt_feature"].shape[0]
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = compute_losses(model, batch, config)
        loss.backward()
        optimizer.step()

        accumulate_metrics(metric_sums, metrics, batch_size)
        total_count += batch_size

    return average_metrics(metric_sums, total_count)


def evaluate(model, val_loader, device, config: LossConfig) -> Dict[str, float]:
    model.eval()
    metric_sums: Dict[str, float] = {}
    total_count = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = move_batch_to_device(batch, device)
            batch_size = batch["neutral_prompt_feature"].shape[0]
            _, metrics = compute_losses(model, batch, config)
            accumulate_metrics(metric_sums, metrics, batch_size)
            total_count += batch_size

    return average_metrics(metric_sums, total_count)


def load_state_dict_flexible(model, checkpoint_path: str, device: torch.device, strict: bool = True):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    model_is_parallel = isinstance(model, torch.nn.DataParallel)
    has_module_prefix = any(key.startswith("module.") for key in state_dict.keys())
    if model_is_parallel and not has_module_prefix:
        state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    elif not model_is_parallel and has_module_prefix:
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}

    load_result = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        print(f"Loaded checkpoint with missing keys: {load_result.missing_keys}")
        print(f"Loaded checkpoint with unexpected keys: {load_result.unexpected_keys}")


def save_model(model, path: str):
    torch.save(model.state_dict(), path)


def format_metrics(prefix: str, metrics: Dict[str, float]) -> str:
    parts = [f"{prefix}_{key}: {value:.4f}" for key, value in sorted(metrics.items())]
    return ", ".join(parts)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="Global batch size across all GPUs.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--load_model", type=str, default=None)
    parser.add_argument("--allow_partial_load", action="store_true")
    parser.add_argument("--device_cuda", type=str, default="0,1", help="Comma-separated CUDA ids, e.g. 0,1.")
    parser.add_argument("--scale_factor", type=float, default=1.0)
    parser.add_argument("--wandb_name", type=str, default="your experiment name")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--enable_density", type=str2bool, default=False)
    parser.add_argument("--density_min", type=float, default=1e-6)
    parser.add_argument("--density_max_weight", type=float, default=0.0)
    parser.add_argument("--data_cache_path", type=str, default="./data/data-cache.pt")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", type=str2bool, default=True)
    parser.add_argument("--enable_easa", action="store_true")
    parser.add_argument("--easa_hidden_dim", type=int, default=64)
    parser.add_argument("--easa_init_bias", type=float, default=2.0)
    parser.add_argument("--easa_anchor_weight", type=float, default=0.1)
    parser.add_argument("--easa_smooth_weight", type=float, default=0.01)
    parser.add_argument("--easa_tau_negative", type=float, default=0.95)
    parser.add_argument("--easa_tau_positive", type=float, default=0.85)
    return parser.parse_args()


def main():
    args = parse_args()
    device, device_ids = setup_device(args.device_cuda)
    n_gpu = len(device_ids)

    if args.use_wandb:
        wandb.init(project="emoticrafter", name=f"{args.wandb_name}", config=vars(args))

    print(f"args\n{args}")
    print(f"device: {device}, data_parallel_gpus: {device_ids}, global_batch_size: {args.batch_size}")

    data = torch.load(args.data_cache_path, map_location="cpu")
    arousal_values = [tensor_scalar(item["arousal"]) for item in data]
    valence_values = [tensor_scalar(item["valence"]) for item in data]
    density_values = get_density(arousal_values, valence_values)
    for index, density in enumerate(density_values):
        data[index]["density"] = float(density)

    train_data, val_data = train_test_split(data, test_size=0.01, random_state=42)
    train_dataset = EmotionDataset(train_data)
    val_dataset = EmotionDataset(val_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and device.type == "cuda",
    )

    config = GPT2Config.from_pretrained("./config")
    model = EmotionInjectionTransformer(
        config,
        final_out_type="Linear+LN",
        use_easa=args.enable_easa,
        easa_hidden_dim=args.easa_hidden_dim,
        easa_init_bias=args.easa_init_bias,
    ).to(device)
    if n_gpu > 1:
        print(f"Using DataParallel on GPUs: {device_ids}")
        model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
    if args.load_model:
        load_state_dict_flexible(model, args.load_model, device, strict=not args.allow_partial_load)
    model.to(device)

    loss_config = LossConfig(
        scale_factor=args.scale_factor,
        enable_density=args.enable_density,
        density_min=args.density_min,
        density_max_weight=args.density_max_weight,
        enable_easa=args.enable_easa,
        easa_anchor_weight=args.easa_anchor_weight,
        easa_smooth_weight=args.easa_smooth_weight,
        easa_tau_negative=args.easa_tau_negative,
        easa_tau_positive=args.easa_tau_positive,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_val_loss = float("inf")
    best_val_emo_loss = float("inf")
    os.makedirs(args.save_dir, exist_ok=True)
    best_model_path = os.path.join(args.save_dir, "best_model.pth")
    best_model_path_emo = os.path.join(args.save_dir, "best_model_emo.pth")

    print("Preparation Done!")

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, loss_config)
        val_metrics = evaluate(model, val_loader, device, loss_config)

        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                **{f"train/{key}": value for key, value in train_metrics.items()},
                **{f"val/{key}": value for key, value in val_metrics.items()},
            })

        print(
            f"Epoch {epoch + 1}/{args.epochs}, "
            f"{format_metrics('train', train_metrics)}, "
            f"{format_metrics('val', val_metrics)}"
        )

        if (epoch + 1) % 50 == 0:
            save_model(model, os.path.join(args.save_dir, f"model_epoch_{epoch + 1}.pth"))

        val_loss = val_metrics.get("loss", float("inf"))
        val_emo_loss = val_metrics.get("loss_emo", float("inf"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, best_model_path)
            print(f"New best model saved with validation total loss: {best_val_loss:.4f}")
        if val_emo_loss < best_val_emo_loss:
            best_val_emo_loss = val_emo_loss
            save_model(model, best_model_path_emo)
            print(f"New best model saved with validation emotion loss: {best_val_emo_loss:.4f}")

    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
