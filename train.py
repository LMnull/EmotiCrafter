import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from transformers import GPT2Config
import wandb
import argparse
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity
from model import EmotionInjectionTransformer
import numpy as np


def _emotion_scalar_for_density(t):
    t = torch.as_tensor(t).float().flatten()
    if t.numel() == 1:
        return float(t.item())
    return float(torch.argmax(t).item())


def get_density(alist, vlist):
    data = np.vstack([alist.T[0], vlist.T[0]]).T
    kde = KernelDensity(kernel='gaussian', bandwidth="silverman")
    kde.fit(data)
    log_density = kde.score_samples(data)
    density = np.exp(log_density)
    return density


class EmotionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sample = {
            'neutral_prompt_feature': item['neutral_prompt_feature'],
            'arousal': item['arousal'],
            'valence': item['valence'],
            'emotional_prompt_feature': item['emotional_prompt_feature'],
            'density': torch.FloatTensor([item['density']]),
        }
        if 'neutral_pooled_prompt_feature' in item and 'emotional_pooled_prompt_feature' in item:
            sample['neutral_pooled_prompt_feature'] = item['neutral_pooled_prompt_feature']
            sample['emotional_pooled_prompt_feature'] = item['emotional_pooled_prompt_feature']
        return sample


def _scaled_target(source_feature, target_feature, alpha):
    return (target_feature - source_feature) * alpha + source_feature


def _mse_with_optional_density(criterion, prediction, target, density=None):
    if density is None:
        return criterion(prediction, target)
    weight = 1 / density
    while weight.dim() < prediction.dim():
        weight = weight.unsqueeze(-1)
    return criterion(prediction * weight, target * weight)


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if not distributed:
        return False, 0, 0, 1

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, local_rank, rank, world_size


def cleanup_distributed(distributed):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def reduce_average(total_loss, total_steps, device, distributed):
    stats = torch.tensor([total_loss, total_steps], dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    if stats[1].item() == 0:
        return 0.0
    return (stats[0] / stats[1]).item()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def load_state_dict_flexible(model, checkpoint_path, device):
    state_dict = torch.load(checkpoint_path, map_location=device)
    if all(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)


def resolve_amp_dtype(amp_dtype, device, rank):
    if device.type != 'cuda' or amp_dtype == 'none':
        return None
    if amp_dtype == 'bf16':
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        if is_main_process(rank):
            print("bf16 is not supported on this GPU; falling back to fp16 AMP.")
        return torch.float16
    if amp_dtype == 'fp16':
        return torch.float16
    raise ValueError(f"Unsupported amp dtype: {amp_dtype}")


def autocast_context(device, amp_dtype):
    return torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None)


def create_grad_scaler(amp_dtype):
    return torch.cuda.amp.GradScaler(enabled=amp_dtype == torch.float16)


def train(
        model,
        train_loader,
        optimizer,
        criterion,
        device,
        alpha=1.0,
        enable_density=False,
        distributed=False,
        rank=0,
        amp_dtype=None,
        scaler=None,
        grad_accum_steps=1,
):
    model.train()
    total_loss = 0
    total_steps = 0

    iterator = tqdm(train_loader, desc="Training", disable=not is_main_process(rank))
    optimizer.zero_grad(set_to_none=True)
    for batch in iterator:
        neutral_prompt_feature = batch['neutral_prompt_feature'].to(device, non_blocking=True).to(torch.float32)
        arousal = batch['arousal'].to(device, non_blocking=True)
        valence = batch['valence'].to(device, non_blocking=True)
        emotional_prompt_feature = batch['emotional_prompt_feature'].to(device, non_blocking=True).to(torch.float32)
        density = batch['density'].to(device, non_blocking=True).to(torch.float32)
        neutral_pooled_prompt_feature = batch.get('neutral_pooled_prompt_feature')
        emotional_pooled_prompt_feature = batch.get('emotional_pooled_prompt_feature')
        if neutral_pooled_prompt_feature is not None:
            neutral_pooled_prompt_feature = neutral_pooled_prompt_feature.to(device, non_blocking=True).to(torch.float32)
            emotional_pooled_prompt_feature = emotional_pooled_prompt_feature.to(device, non_blocking=True).to(torch.float32)
        with autocast_context(device, amp_dtype):
            outputs = model(
                inputs_embeds=neutral_prompt_feature,
                pooled_prompt_embeds=neutral_pooled_prompt_feature,
                arousal=arousal,
                valence=valence,
            )
            predicted_emotional_prompt_feature = outputs[0]
            token_target = _scaled_target(neutral_prompt_feature, emotional_prompt_feature, alpha)
            loss = _mse_with_optional_density(
                criterion,
                predicted_emotional_prompt_feature,
                token_target,
                density if enable_density else None,
            )
            if neutral_pooled_prompt_feature is not None and len(outputs) > 1:
                pooled_target = _scaled_target(neutral_pooled_prompt_feature, emotional_pooled_prompt_feature, alpha)
                pooled_loss = _mse_with_optional_density(
                    criterion,
                    outputs[1],
                    pooled_target,
                    density if enable_density else None,
                )
                loss = loss + pooled_loss
        loss_to_backward = loss / grad_accum_steps
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss_to_backward).backward()
        else:
            loss_to_backward.backward()

        should_step = (total_steps + 1) % grad_accum_steps == 0 or (total_steps + 1) == len(train_loader)
        if should_step:
            if scaler is not None and scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        total_steps += 1

    return reduce_average(total_loss, total_steps, device, distributed)


def evaluate(
        model,
        val_loader,
        criterion,
        device,
        alpha=1.0,
        enable_density=False,
        distributed=False,
        rank=0,
        amp_dtype=None,
):
    model.eval()
    total_loss = 0
    total_loss_weight = 0
    total_steps = 0
    total_weight_steps = 0
    with torch.no_grad():
        iterator = tqdm(val_loader, desc="Evaluating", disable=not is_main_process(rank))
        for batch in iterator:
            neutral_prompt_feature = batch['neutral_prompt_feature'].to(device, non_blocking=True).to(torch.float32)
            arousal = batch['arousal'].to(device, non_blocking=True)
            valence = batch['valence'].to(device, non_blocking=True)
            emotional_prompt_feature = batch['emotional_prompt_feature'].to(device, non_blocking=True).to(torch.float32)
            density = batch['density'].to(device, non_blocking=True).to(torch.float32)
            neutral_pooled_prompt_feature = batch.get('neutral_pooled_prompt_feature')
            emotional_pooled_prompt_feature = batch.get('emotional_pooled_prompt_feature')
            if neutral_pooled_prompt_feature is not None:
                neutral_pooled_prompt_feature = neutral_pooled_prompt_feature.to(device, non_blocking=True).to(torch.float32)
                emotional_pooled_prompt_feature = emotional_pooled_prompt_feature.to(device, non_blocking=True).to(torch.float32)

            with autocast_context(device, amp_dtype):
                outputs = model(
                    inputs_embeds=neutral_prompt_feature,
                    pooled_prompt_embeds=neutral_pooled_prompt_feature,
                    arousal=arousal,
                    valence=valence,
                )
                predicted_emotional_prompt_feature = outputs[0]
                token_target = _scaled_target(neutral_prompt_feature, emotional_prompt_feature, alpha)
                if enable_density:
                    loss_weight = _mse_with_optional_density(
                        criterion,
                        predicted_emotional_prompt_feature,
                        token_target,
                        density,
                    )
                    if neutral_pooled_prompt_feature is not None and len(outputs) > 1:
                        pooled_target = _scaled_target(
                            neutral_pooled_prompt_feature,
                            emotional_pooled_prompt_feature,
                            alpha,
                        )
                        loss_weight = loss_weight + _mse_with_optional_density(
                            criterion,
                            outputs[1],
                            pooled_target,
                            density,
                        )
                    total_loss_weight += loss_weight.item()
                    total_weight_steps += 1
                loss = criterion(predicted_emotional_prompt_feature, token_target)
                if neutral_pooled_prompt_feature is not None and len(outputs) > 1:
                    pooled_target = _scaled_target(
                        neutral_pooled_prompt_feature,
                        emotional_pooled_prompt_feature,
                        alpha,
                    )
                    loss = loss + criterion(outputs[1], pooled_target)

            total_loss += loss.item()
            total_steps += 1
    return (
        reduce_average(total_loss, total_steps, device, distributed),
        reduce_average(total_loss_weight, total_weight_steps, device, distributed),
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='Per-GPU micro-batch size when launched with torchrun/DDP.')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--device_cuda', type=str, default="0")
    parser.add_argument('--scale_factor', type=float, default=1.5)
    parser.add_argument('--wandb_name', type=str, default="lfy_emotion")
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--enable_density', action='store_true', default=False)
    parser.add_argument('--data_cache_path', type=str, default="./data/data-cache.pt")
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--grad_accum_steps', type=int, default=8)
    parser.add_argument('--amp_dtype', type=str, default='bf16', choices=['bf16', 'fp16', 'none'])

    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be greater than 0")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad_accum_steps must be greater than 0")

    distributed, local_rank, rank, world_size = setup_distributed()
    main_process = is_main_process(rank)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}" if distributed else f"cuda:{args.device_cuda}")
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    if distributed and not torch.cuda.is_available():
        raise RuntimeError("Distributed training requires CUDA devices.")
    amp_dtype = resolve_amp_dtype(args.amp_dtype, device, rank)

    use_wandb = args.use_wandb and main_process
    if use_wandb:
        wandb.init(project="emoticrafter", name=f"{args.wandb_name}", config=vars(args))
    if main_process:
        print(f"args\n{args}")
        print(f"distributed={distributed}, world_size={world_size}, device={device}")
        effective_batch = args.batch_size * world_size * args.grad_accum_steps
        print(
            f"micro_batch_per_gpu={args.batch_size}, grad_accum_steps={args.grad_accum_steps}, "
            f"effective_batch_size={effective_batch}, amp_dtype={args.amp_dtype}"
        )

    data = torch.load(args.data_cache_path, map_location='cpu')

    alist = np.array([[_emotion_scalar_for_density(item['arousal'])] for item in data])
    vlist = np.array([[_emotion_scalar_for_density(item['valence'])] for item in data])

    den_list = get_density(alist, vlist)
    for index in range(len(data)):
        data[index]['density'] = den_list[index]

    train_data, val_data = train_test_split(data, test_size=0.01, random_state=42)
    train_dataset = EmotionDataset(train_data)
    val_dataset = EmotionDataset(val_data)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
    ) if distributed else None
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    ) if distributed else None
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=train_sampler is None,
                              sampler=train_sampler,
                              num_workers=args.num_workers,
                              pin_memory=pin_memory,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            sampler=val_sampler,
                            num_workers=args.num_workers,
                            pin_memory=pin_memory,
                            persistent_workers=args.num_workers > 0
                            )

    alpha = args.scale_factor

    config = GPT2Config.from_pretrained('./config')
    model = EmotionInjectionTransformer(config, final_out_type="DisentangledDualCondition").to(device)
    if args.load_model:
        load_state_dict_flexible(model, args.load_model, device)
    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()
    scaler = create_grad_scaler(amp_dtype)

    best_val_loss = float('inf')
    best_model_path = os.path.join(args.save_dir, 'best_model.pth')
    best_model_path_weight = os.path.join(args.save_dir, 'best_model_weight.pth')
    if args.enable_density:
        best_val_loss_weight = float('inf')

    if main_process:
        print("Preparation Done!")

    try:
        for epoch in range(args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_loss = train(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                alpha,
                enable_density=args.enable_density,
                distributed=distributed,
                rank=rank,
                amp_dtype=amp_dtype,
                scaler=scaler,
                grad_accum_steps=args.grad_accum_steps,
            )
            val_loss, val_loss_weight = evaluate(
                model,
                val_loader,
                criterion,
                device,
                alpha,
                args.enable_density,
                distributed=distributed,
                rank=rank,
                amp_dtype=amp_dtype,
            )
            if use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_loss_weight': val_loss_weight
                })

            if main_process:
                print(
                    f"Epoch {epoch + 1}/{args.epochs}, Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val loss weight: {val_loss_weight:.4f}"
                )

                # Save model
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir)
                state_dict = unwrap_model(model).state_dict()
                if (epoch + 1) % 50 == 0:
                    torch.save(state_dict, os.path.join(args.save_dir, f'model_epoch_{epoch + 1}.pth'))

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(state_dict, best_model_path)
                    print(f"New best model saved with validation loss: {best_val_loss:.4f}")
                if args.enable_density:
                    if val_loss_weight < best_val_loss_weight:
                        best_val_loss_weight = val_loss_weight
                        torch.save(state_dict, best_model_path_weight)
                        print(f"New best model saved with weighted validation loss: {best_val_loss_weight:.4f}")
    finally:
        if use_wandb:
            wandb.finish()
        cleanup_distributed(distributed)


if __name__ == '__main__':
    main()
