import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "results" / "val_prompt_5x5"
DEFAULT_MODEL_PATH = Path("/root/shared-nvme/model/clip-vit-large-patch14")
DEFAULT_LOG_PATH = PROJECT_ROOT / "log.txt"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
PROMPT_NAME_RE = re.compile(r"^(?P<prompt>.+)_v-?\d+(?:\.\d+)?_a-?\d+(?:\.\d+)?$")


class CLIPScore:
    """
    CLIPScore for image-text alignment.

    The per-sample score follows:
        score_weight * max(0, cos(text_feature, image_feature))
    """

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        device: Optional[str] = None,
        text_prefix: str = "",
        score_weight: float = 100.0,
        output_scale: float = 1.0,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(str(model_path)).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(str(model_path))
        self.model.eval()
        self.text_prefix = text_prefix
        self.score_weight = score_weight
        self.output_scale = output_scale

    def compute_image_features(
        self,
        images: Union[str, Path, Image.Image, Sequence[Union[str, Path, Image.Image]]],
    ):
        if isinstance(images, (str, Path, Image.Image)):
            images = [images]

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

    def compute_text_features(self, texts: Union[str, Sequence[str]]):
        if isinstance(texts, str):
            texts = [texts]
        texts = [f"{self.text_prefix}{text}" if self.text_prefix else text for text in texts]

        inputs = self.processor(
            text=list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = F.normalize(text_features, dim=-1)

        return text_features

    def score_pairs(
        self,
        images: Sequence[Union[str, Path, Image.Image]],
        texts: Sequence[str],
    ) -> np.ndarray:
        if len(images) != len(texts):
            raise ValueError("images and texts must have the same length for paired CLIPScore")

        image_features = self.compute_image_features(images)
        text_features = self.compute_text_features(texts)
        similarities = (image_features * text_features).sum(dim=-1)
        scores = torch.clamp(similarities, min=0) * self.score_weight * self.output_scale
        return scores.detach().cpu().numpy()

    def score(
        self,
        images: Union[str, Path, Image.Image, Sequence[Union[str, Path, Image.Image]]],
        texts: Union[str, Sequence[str]],
        reduction: str = "mean",
    ):
        if isinstance(images, (str, Path, Image.Image)):
            image_list = [images]
        else:
            image_list = list(images)

        text_list = [texts] if isinstance(texts, str) else list(texts)
        if len(image_list) == len(text_list):
            scores = self.score_pairs(image_list, text_list)
        else:
            image_features = self.compute_image_features(image_list)
            text_features = self.compute_text_features(text_list)
            scores = (
                torch.clamp(image_features @ text_features.T, min=0)
                * self.score_weight
                * self.output_scale
            ).detach().cpu().numpy()

        if reduction == "mean":
            return float(np.mean(scores))
        if reduction == "sum":
            return float(np.sum(scores))
        if reduction == "none":
            return scores
        raise ValueError("reduction must be one of: mean, sum, none")


def parse_prompt_from_image_name(image_path: Union[str, Path]) -> str:
    stem = Path(image_path).stem
    match = PROMPT_NAME_RE.match(stem)
    if match is None:
        raise ValueError(f"Cannot parse prompt from image name: {Path(image_path).name}")
    return match.group("prompt")


def iter_image_prompt_pairs(image_dir: Union[str, Path]) -> Iterable[Tuple[Path, str]]:
    image_dir = Path(image_dir)
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        yield image_path, parse_prompt_from_image_name(image_path)


def chunked(items: Sequence[Tuple[Path, str]], batch_size: int):
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
    text_prefix: str = "",
    score_weight: float = 100.0,
    output_scale: float = 1.0,
):
    if batch_size < 1:
        raise ValueError("batch_size must be greater than 0")

    pairs = list(iter_image_prompt_pairs(image_dir))
    if limit is not None:
        pairs = pairs[:limit]
    if not pairs:
        raise ValueError(f"No valid images found in {image_dir}")

    metric = CLIPScore(
        model_path=model_path,
        device=device,
        text_prefix=text_prefix,
        score_weight=score_weight,
        output_scale=output_scale,
    )
    all_scores: List[float] = []

    for batch_index, batch in enumerate(chunked(pairs, batch_size), start=1):
        image_paths = [image_path for image_path, _ in batch]
        prompts = [prompt for _, prompt in batch]
        scores = metric.score_pairs(image_paths, prompts)
        all_scores.extend(float(score) for score in scores)
        print(f"batch {batch_index}: processed {len(all_scores)}/{len(pairs)}", flush=True)

    return np.asarray(all_scores, dtype=np.float64), pairs


def write_log(
    log_path: Union[str, Path],
    scores: np.ndarray,
    append: bool,
    model_path: Union[str, Path],
    text_prefix: str,
    score_weight: float,
    output_scale: float,
):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"CLIPScore model_path: {model_path}",
        f"CLIPScore text_prefix: {text_prefix!r}",
        f"CLIPScore score_weight: {score_weight}",
        f"CLIPScore output_scale: {output_scale}",
        f"CLIPScore: {format_mean_std(scores)}",
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
    parser.add_argument("--text_prefix", type=str, default="")
    parser.add_argument("--score_weight", type=float, default=100.0)
    parser.add_argument("--output_scale", type=float, default=1.0)
    parser.add_argument("--overwrite_log", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    scores, _ = evaluate_directory(
        image_dir=args.image_dir,
        model_path=args.model_path,
        device=args.device,
        batch_size=args.batch_size,
        limit=args.limit,
        text_prefix=args.text_prefix,
        score_weight=args.score_weight,
        output_scale=args.output_scale,
    )
    used_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    write_log(
        log_path=args.log_path,
        scores=scores,
        append=not args.overwrite_log,
        model_path=args.model_path,
        text_prefix=args.text_prefix,
        score_weight=args.score_weight,
        output_scale=args.output_scale,
    )
