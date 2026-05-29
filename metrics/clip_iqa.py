import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "results" / "val_prompt_5x5"
DEFAULT_MODEL_PATH = Path("/root/shared-nvme/model/clip-vit-large-patch14")
DEFAULT_LOG_PATH = PROJECT_ROOT / "log.txt"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class CLIPIQA:
    """
    CLIP-IQA style no-reference image quality score.

    Scores are in the same 0-1 range as the original implementation in this repo.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(str(model_path)).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(str(model_path))
        self.model.eval()

        self.quality_levels = [
            "excellent quality",
            "good quality",
            "fair quality",
            "poor quality",
            "bad quality",
        ]
        self.quality_scores = torch.tensor([100, 75, 50, 25, 0], dtype=torch.float32, device=self.device)
        self.anchor_features = self.encode_text_prompts(self.quality_levels)

    def encode_text_prompts(self, prompts: Sequence[str]) -> torch.Tensor:
        inputs = self.processor(
            text=list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = F.normalize(text_features, dim=-1)
        return text_features

    def encode_images(self, images: Sequence[Union[str, Path, Image.Image]]) -> torch.Tensor:
        pil_images = []
        for image in images:
            if isinstance(image, Image.Image):
                pil_images.append(image.convert("RGB"))
            else:
                with Image.open(image) as opened_image:
                    pil_images.append(opened_image.convert("RGB"))

        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = F.normalize(image_features, dim=-1)
        return image_features

    def compute_quality_scores(self, images: Sequence[Union[str, Path, Image.Image]]) -> np.ndarray:
        image_features = self.encode_images(images)
        similarities = image_features @ self.anchor_features.T
        weights = F.softmax(similarities * 10, dim=-1)
        scores = (weights * self.quality_scores).sum(dim=-1) / 100
        return scores.detach().cpu().numpy()

    def compute_quality_score(self, image: Union[str, Path, Image.Image]) -> float:
        return float(self.compute_quality_scores([image])[0])


def iter_image_paths(image_dir: Union[str, Path]) -> List[Path]:
    image_dir = Path(image_dir)
    return sorted(
        image_path
        for image_path in image_dir.iterdir()
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
    )


def chunked(items: Sequence[Path], batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index:index + batch_size]


def format_mean_std(values: Sequence[float]) -> str:
    values = np.asarray(values, dtype=np.float64)
    return f"{values.mean():.3f} \u00b1 {values.std(ddof=0):.3f}"


def evaluate_directory(
    image_dir: Union[str, Path],
    model_path: Union[str, Path],
    device: Optional[str],
    batch_size: int,
    limit: Optional[int] = None,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be greater than 0")

    image_paths = iter_image_paths(image_dir)
    if limit is not None:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise ValueError(f"No valid images found in {image_dir}")

    metric = CLIPIQA(model_path=model_path, device=device)
    all_scores: List[float] = []

    for batch_index, batch in enumerate(chunked(image_paths, batch_size), start=1):
        scores = metric.compute_quality_scores(batch)
        all_scores.extend(float(score) for score in scores)
        print(f"batch {batch_index}: processed {len(all_scores)}/{len(image_paths)}", flush=True)

    return np.asarray(all_scores, dtype=np.float64)


def write_log(
    log_path: Union[str, Path],
    scores: np.ndarray,
    append: bool,
):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"CLIP-IQA: {format_mean_std(scores)}",
        f"num_images: {len(scores)}",
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
    parser.add_argument("--log_path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite_log", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scores = evaluate_directory(
        image_dir=args.image_dir,
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    used_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    write_log(
        log_path=args.log_path,
        scores=scores,
        append=not args.overwrite_log,
    )
