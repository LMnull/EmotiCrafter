import argparse
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "results" / "val_prompt_5x5"
DEFAULT_MODEL_PATH = Path("/root/shared-nvme/model/clip-vit-large-patch14")
DEFAULT_LOG_PATH = PROJECT_ROOT / "log.txt"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CLIPIQA_BACKENDS = Literal["auto", "transformers", "openai_clip"]


class CLIPIQA:
    """
    CLIP-IQA no-reference image quality score.

    This follows the official CLIP-IQA fixed-prompt formulation:
    compute similarities to a positive/negative prompt pair and return the
    softmax probability of the positive prompt.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        device: Optional[str] = None,
        prompt_pair: Tuple[str, str] = ("Good photo.", "Bad photo."),
        logit_scale: Literal["learned", "100"] = "learned",
        backend: CLIPIQA_BACKENDS = "auto",
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if len(prompt_pair) != 2:
            raise ValueError("prompt_pair must contain exactly two prompts: positive and negative.")
        self.model_path = Path(model_path)
        self.prompt_pair = prompt_pair
        self.logit_scale = logit_scale
        self.backend = self.resolve_backend(backend, self.model_path)

        if self.backend == "openai_clip":
            self.init_openai_clip(self.model_path)
        elif self.backend == "transformers":
            self.init_transformers_clip(self.model_path)
        else:
            raise ValueError(f"Unsupported CLIP-IQA backend: {self.backend}")

        self.anchor_features = self.encode_text_prompts(self.prompt_pair)

    @staticmethod
    def resolve_backend(backend: CLIPIQA_BACKENDS, model_path: Path) -> str:
        if backend != "auto":
            return backend
        return "openai_clip" if model_path.is_file() and model_path.suffix == ".pt" else "transformers"

    def init_transformers_clip(self, model_path: Path):
        from transformers import CLIPModel, CLIPProcessor

        self.model = CLIPModel.from_pretrained(str(model_path)).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(str(model_path))
        self.model.eval()

    def init_openai_clip(self, model_path: Path):
        try:
            import clip
        except ImportError as exc:
            raise ImportError(
                "OpenAI CLIP is required for RN50.pt CLIP-IQA. Install it with:\n"
                "pip install git+https://github.com/openai/CLIP.git"
            ) from exc

        self.clip = clip
        self.model, self.preprocess = clip.load(str(model_path), device=self.device)
        self.model.eval()
        self.tokenized_prompts = clip.tokenize(list(self.prompt_pair)).to(self.device)

    def encode_text_prompts(self, prompts: Sequence[str]) -> torch.Tensor:
        if self.backend == "openai_clip":
            with torch.no_grad():
                text_features = self.model.encode_text(self.tokenized_prompts)
                text_features = F.normalize(text_features.float(), dim=-1)
            return text_features

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

        if self.backend == "openai_clip":
            image_tensor = torch.stack([self.preprocess(image) for image in pil_images]).to(self.device)
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                image_features = F.normalize(image_features.float(), dim=-1)
            return image_features

        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            image_features = F.normalize(image_features, dim=-1)
        return image_features

    def compute_quality_scores(self, images: Sequence[Union[str, Path, Image.Image]]) -> np.ndarray:
        image_features = self.encode_images(images)
        similarities = image_features @ self.anchor_features.T
        if self.logit_scale == "learned":
            scale = self.model.logit_scale.exp().to(similarities.device)
        elif self.logit_scale == "100":
            scale = 100.0
        else:
            raise ValueError("logit_scale must be either 'learned' or '100'")
        scores = F.softmax(similarities * scale, dim=-1)[:, 0]
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
    positive_prompt: str = "Good photo.",
    negative_prompt: str = "Bad photo.",
    logit_scale: Literal["learned", "100"] = "learned",
    backend: CLIPIQA_BACKENDS = "auto",
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be greater than 0")

    image_paths = iter_image_paths(image_dir)
    if limit is not None:
        image_paths = image_paths[:limit]
    if not image_paths:
        raise ValueError(f"No valid images found in {image_dir}")

    metric = CLIPIQA(
        model_path=model_path,
        device=device,
        prompt_pair=(positive_prompt, negative_prompt),
        logit_scale=logit_scale,
        backend=backend,
    )
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
    parser.add_argument("--positive_prompt", type=str, default="Good photo.")
    parser.add_argument("--negative_prompt", type=str, default="Bad photo.")
    parser.add_argument("--logit_scale", choices=["learned", "100"], default="learned")
    parser.add_argument("--backend", choices=["auto", "transformers", "openai_clip"], default="auto")
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
        positive_prompt=args.positive_prompt,
        negative_prompt=args.negative_prompt,
        logit_scale=args.logit_scale,
        backend=args.backend,
    )
    used_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    write_log(
        log_path=args.log_path,
        scores=scores,
        append=not args.overwrite_log,
    )
