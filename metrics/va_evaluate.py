import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "results" / "val_prompt_5x5"
DEFAULT_MODEL_PATH = Path("/root/shared-nvme/model/clip-vit-base-patch32")
DEFAULT_AROUSAL_CKPT = METRICS_DIR / "arousal1_CLIP_lr=0.001_loss=MSELoss_sc=test_cuda-1.pth"
DEFAULT_VALENCE_CKPT = METRICS_DIR / "valence1_CLIP_lr=0.0001_loss=MSELoss_sc=test_cuda.pth"
DEFAULT_LOG_PATH = PROJECT_ROOT / "log.txt"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VA_NAME_RE = re.compile(r"^(?P<prompt>.+)_v(?P<valence>-?\d+(?:\.\d+)?)_a(?P<arousal>-?\d+(?:\.\d+)?)$")


class CLIPRegressor1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=512, out_features=1, bias=True),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.classifier(x)


class va_predictor(torch.nn.Module):
    def __init__(
        self,
        model_path: Union[str, Path],
        device: str,
        arousal_ckpt: Union[str, Path] = DEFAULT_AROUSAL_CKPT,
        valence_ckpt: Union[str, Path] = DEFAULT_VALENCE_CKPT,
    ):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(str(model_path)).to(device)
        self.processor = CLIPProcessor.from_pretrained(str(model_path))
        self.device = device
        self.ar = CLIPRegressor1().to(device)
        self.vr = CLIPRegressor1().to(device)
        self.ar.load_state_dict(torch.load(arousal_ckpt, map_location=device))
        self.vr.load_state_dict(torch.load(valence_ckpt, map_location=device))
        self.eval()

    def forward(self, images):
        if isinstance(images, Image.Image):
            images = [images]

        pil_images = []
        for image in images:
            if isinstance(image, Image.Image):
                pil_images.append(image.convert("RGB"))
            else:
                with Image.open(image) as opened_image:
                    pil_images.append(opened_image.convert("RGB"))

        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            pred_valence = self.vr(image_features) * 6 - 3
            pred_arousal = self.ar(image_features) * 6 - 3
        return pred_valence.squeeze(-1), pred_arousal.squeeze(-1)


def parse_va_from_image_name(image_path: Union[str, Path]) -> Tuple[float, float]:
    stem = Path(image_path).stem
    match = VA_NAME_RE.match(stem)
    if match is None:
        raise ValueError(f"Cannot parse valence/arousal from image name: {Path(image_path).name}")
    return float(match.group("valence")), float(match.group("arousal"))


def iter_image_va_pairs(image_dir: Union[str, Path]) -> List[Tuple[Path, float, float]]:
    image_dir = Path(image_dir)
    pairs = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        valence, arousal = parse_va_from_image_name(image_path)
        pairs.append((image_path, valence, arousal))
    return pairs


def chunked(items: Sequence[Tuple[Path, float, float]], batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index:index + batch_size]


def format_mean_std(values: Sequence[float]) -> str:
    values = np.asarray(values, dtype=np.float64)
    return f"{values.mean():.3f} \u00b1 {values.std(ddof=0):.3f}"


def evaluate_directory(
    image_dir: Union[str, Path],
    model_path: Union[str, Path],
    arousal_ckpt: Union[str, Path],
    valence_ckpt: Union[str, Path],
    device: Optional[str],
    batch_size: int,
    limit: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    if batch_size < 1:
        raise ValueError("batch_size must be greater than 0")

    used_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pairs = iter_image_va_pairs(image_dir)
    if limit is not None:
        pairs = pairs[:limit]
    if not pairs:
        raise ValueError(f"No valid images found in {image_dir}")

    predictor = va_predictor(
        model_path=model_path,
        device=used_device,
        arousal_ckpt=arousal_ckpt,
        valence_ckpt=valence_ckpt,
    )

    target_valences: List[float] = []
    target_arousals: List[float] = []
    pred_valences: List[float] = []
    pred_arousals: List[float] = []

    for batch_index, batch in enumerate(chunked(pairs, batch_size), start=1):
        image_paths = [image_path for image_path, _, _ in batch]
        batch_target_valences = np.asarray([valence for _, valence, _ in batch], dtype=np.float64)
        batch_target_arousals = np.asarray([arousal for _, _, arousal in batch], dtype=np.float64)
        pred_valence, pred_arousal = predictor(image_paths)

        target_valences.extend(batch_target_valences.tolist())
        target_arousals.extend(batch_target_arousals.tolist())
        pred_valences.extend(pred_valence.detach().cpu().numpy().astype(np.float64).tolist())
        pred_arousals.extend(pred_arousal.detach().cpu().numpy().astype(np.float64).tolist())
        print(f"batch {batch_index}: processed {len(pred_valences)}/{len(pairs)}", flush=True)

    target_valence_array = np.asarray(target_valences, dtype=np.float64)
    target_arousal_array = np.asarray(target_arousals, dtype=np.float64)
    pred_valence_array = np.asarray(pred_valences, dtype=np.float64)
    pred_arousal_array = np.asarray(pred_arousals, dtype=np.float64)

    return {
        "target_valence": target_valence_array,
        "target_arousal": target_arousal_array,
        "pred_valence": pred_valence_array,
        "pred_arousal": pred_arousal_array,
        "valence_abs_error": np.abs(pred_valence_array - target_valence_array),
        "arousal_abs_error": np.abs(pred_arousal_array - target_arousal_array),
    }


def write_log(
    log_path: Union[str, Path],
    results: Dict[str, np.ndarray],
    append: bool,
):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    num_images = len(results["pred_valence"])

    lines = [
        f"VA pred_valence: {format_mean_std(results['pred_valence'])}",
        f"VA pred_arousal: {format_mean_std(results['pred_arousal'])}",
        f"VA valence_abs_error: {format_mean_std(results['valence_abs_error'])}",
        f"VA arousal_abs_error: {format_mean_std(results['arousal_abs_error'])}",
        f"num_images: {num_images}",
    ]
    log_text = "\n".join(lines) + "\n"
    if append and log_path.exists() and log_path.stat().st_size > 0:
        log_text = "\n" + log_text
    mode = "a" if append else "w"
    with log_path.open(mode, encoding="utf-8") as file:
        file.write(log_text)
    print(log_text, end="")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--arousal_ckpt", type=Path, default=DEFAULT_AROUSAL_CKPT)
    parser.add_argument("--valence_ckpt", type=Path, default=DEFAULT_VALENCE_CKPT)
    parser.add_argument("--log_path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite_log", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    used_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    results = evaluate_directory(
        image_dir=args.image_dir,
        model_path=args.model_path,
        arousal_ckpt=args.arousal_ckpt,
        valence_ckpt=args.valence_ckpt,
        device=used_device,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    write_log(
        log_path=args.log_path,
        results=results,
        append=not args.overwrite_log,
    )
