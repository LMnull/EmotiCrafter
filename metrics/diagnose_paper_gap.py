import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_IMAGE_DIR = PROJECT_ROOT / "results" / "val_prompt_5x5"
DEFAULT_CLIP_MODEL_PATH = Path("/root/shared-nvme/model/clip-vit-base-patch32")
DEFAULT_CLIP_IQA_MODEL_PATH = Path("/root/shared-nvme/model/RN50.pt")
DEFAULT_LOG_PATH = PROJECT_ROOT / "diagnostics_log.txt"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
IMAGE_NAME_RE = re.compile(
    r"^(?P<prompt>.+)_v(?P<valence>-?\d+(?:\.\d+)?)_a(?P<arousal>-?\d+(?:\.\d+)?)$"
)


@dataclass
class ImageRecord:
    path: Path
    prompt: str
    valence: float
    arousal: float
    clip_score: Optional[float] = None
    clip_iqa_score: Optional[float] = None


def parse_image_record(image_path: Path) -> ImageRecord:
    match = IMAGE_NAME_RE.match(image_path.stem)
    if match is None:
        raise ValueError(f"Cannot parse prompt/v/a from image name: {image_path.name}")

    return ImageRecord(
        path=image_path,
        prompt=match.group("prompt"),
        valence=float(match.group("valence")),
        arousal=float(match.group("arousal")),
    )


def load_records(image_dir: Path, limit: Optional[int] = None) -> List[ImageRecord]:
    records = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        records.append(parse_image_record(image_path))

    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError(f"No valid images found in {image_dir}")
    return records


def chunked(items: Sequence, batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index:index + batch_size]


def mean_std(values: Sequence[float]) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def format_mean_std(values: Sequence[float]) -> str:
    if not values:
        return "nan \u00b1 nan (n=0)"
    mean, std = mean_std(values)
    return f"{mean:.4f} \u00b1 {std:.4f} (n={len(values)})"


def sorted_unique(values: Sequence[float]) -> List[float]:
    return sorted(set(float(value) for value in values))


def format_va_value(value: float) -> str:
    return f"{value:g}"


def compute_clip_scores(
    records: Sequence[ImageRecord],
    model_path: Path,
    device: Optional[str],
    batch_size: int,
):
    from metrics.clip_score import CLIPScore

    metric = CLIPScore(model_path=model_path, device=device)
    for batch_index, batch in enumerate(chunked(records, batch_size), start=1):
        image_paths = [record.path for record in batch]
        prompts = [record.prompt for record in batch]
        scores = metric.score_pairs(image_paths, prompts)
        for record, score in zip(batch, scores):
            record.clip_score = float(score)
        done = min(batch_index * batch_size, len(records))
        print(f"CLIPScore batch {batch_index}: processed {done}/{len(records)}", flush=True)


def compute_clip_iqa_scores(
    records: Sequence[ImageRecord],
    model_path: Path,
    device: str,
    batch_size: int,
    positive_prompt: str,
    negative_prompt: str,
    logit_scale: str,
    backend: str,
):
    import torch
    from metrics.clip_iqa import CLIPIQA

    metric = CLIPIQA(
        model_path=model_path,
        device=device,
        prompt_pair=(positive_prompt, negative_prompt),
        logit_scale=logit_scale,
        backend=backend,
    )
    with torch.no_grad():
        for batch_index, batch in enumerate(chunked(records, batch_size), start=1):
            image_paths = [record.path for record in batch]
            scores = metric.compute_quality_scores(image_paths)
            for record, score in zip(batch, scores):
                record.clip_iqa_score = float(score)
            done = min(batch_index * batch_size, len(records))
            print(f"CLIP-IQA batch {batch_index}: processed {done}/{len(records)}", flush=True)


def collect_scores(records: Sequence[ImageRecord], score_name: str) -> List[float]:
    values = []
    for record in records:
        value = getattr(record, score_name)
        if value is not None:
            values.append(float(value))
    return values


def scores_for(records: Sequence[ImageRecord], score_name: str, valence=None, arousal=None) -> List[float]:
    values = []
    for record in records:
        if valence is not None and record.valence != valence:
            continue
        if arousal is not None and record.arousal != arousal:
            continue
        value = getattr(record, score_name)
        if value is not None:
            values.append(float(value))
    return values


def add_group_summary(lines: List[str], records: Sequence[ImageRecord], score_name: str, metric_label: str):
    valences = sorted_unique([record.valence for record in records])
    arousals = sorted_unique([record.arousal for record in records])

    lines.append(f"{metric_label} overall: {format_mean_std(collect_scores(records, score_name))}")
    lines.append("")
    lines.append(f"{metric_label} by valence:")
    for valence in valences:
        values = scores_for(records, score_name, valence=valence)
        lines.append(f"  v={format_va_value(valence)}: {format_mean_std(values)}")

    lines.append("")
    lines.append(f"{metric_label} by arousal:")
    for arousal in arousals:
        values = scores_for(records, score_name, arousal=arousal)
        lines.append(f"  a={format_va_value(arousal)}: {format_mean_std(values)}")

    lines.append("")
    lines.append(f"{metric_label} by V-A grid:")
    header = ["a\\v"] + [format_va_value(valence) for valence in valences]
    lines.append("\t".join(header))
    for arousal in arousals:
        row = [format_va_value(arousal)]
        for valence in valences:
            values = scores_for(records, score_name, valence=valence, arousal=arousal)
            row.append(format_mean_std(values))
        lines.append("\t".join(row))

    neutral_values = scores_for(records, score_name, valence=0.0, arousal=0.0)
    lines.append("")
    lines.append(f"{metric_label} neutral point (v=0, a=0): {format_mean_std(neutral_values)}")


def build_report(
    records: Sequence[ImageRecord],
    image_dir: Path,
    clip_model_path: Path,
    clip_iqa_model_path: Path,
    clip_iqa_backend: str,
    clip_iqa_positive_prompt: str,
    clip_iqa_negative_prompt: str,
    clip_iqa_logit_scale: str,
    device: str,
) -> str:
    lines = [
        "EmotiCrafter paper-gap diagnostics",
        f"num_images: {len(records)}",
        f"input_dir: {image_dir}",
        f"clip_model_path: {clip_model_path}",
        f"clip_iqa_model_path: {clip_iqa_model_path}",
        f"clip_iqa_backend: {clip_iqa_backend}",
        f"clip_iqa_positive_prompt: {clip_iqa_positive_prompt}",
        f"clip_iqa_negative_prompt: {clip_iqa_negative_prompt}",
        f"clip_iqa_logit_scale: {clip_iqa_logit_scale}",
        f"device: {device}",
        "",
    ]

    if collect_scores(records, "clip_score"):
        add_group_summary(lines, records, "clip_score", "CLIPScore")
        lines.append("")

    if collect_scores(records, "clip_iqa_score"):
        add_group_summary(lines, records, "clip_iqa_score", "Official CLIP-IQA")

    return "\n".join(lines) + "\n"


def write_report(log_path: Path, report: str):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(report, encoding="utf-8")
    print(report)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--clip_model_path", type=Path, default=DEFAULT_CLIP_MODEL_PATH)
    parser.add_argument("--log_path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip_clipscore", action="store_true")
    parser.add_argument("--skip_clip_iqa", "--skip_pyiqa", action="store_true", dest="skip_clip_iqa")
    parser.add_argument("--clip_iqa_model_path", type=Path, default=DEFAULT_CLIP_IQA_MODEL_PATH)
    parser.add_argument("--clip_iqa_backend", choices=["auto", "transformers", "openai_clip"], default="auto")
    parser.add_argument("--clip_iqa_positive_prompt", type=str, default="Good photo.")
    parser.add_argument("--clip_iqa_negative_prompt", type=str, default="Bad photo.")
    parser.add_argument("--clip_iqa_logit_scale", choices=["learned", "100"], default="learned")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch_size must be greater than 0")

    if args.device:
        device = args.device
    else:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    records = load_records(args.image_dir, limit=args.limit)

    if not args.skip_clipscore:
        compute_clip_scores(
            records=records,
            model_path=args.clip_model_path,
            device=device,
            batch_size=args.batch_size,
        )

    if not args.skip_clip_iqa:
        compute_clip_iqa_scores(
            records=records,
            model_path=args.clip_iqa_model_path,
            device=device,
            batch_size=args.batch_size,
            positive_prompt=args.clip_iqa_positive_prompt,
            negative_prompt=args.clip_iqa_negative_prompt,
            logit_scale=args.clip_iqa_logit_scale,
            backend=args.clip_iqa_backend,
        )

    report = build_report(
        records=records,
        image_dir=args.image_dir,
        clip_model_path=args.clip_model_path,
        clip_iqa_model_path=args.clip_iqa_model_path,
        clip_iqa_backend=args.clip_iqa_backend,
        clip_iqa_positive_prompt=args.clip_iqa_positive_prompt,
        clip_iqa_negative_prompt=args.clip_iqa_negative_prompt,
        clip_iqa_logit_scale=args.clip_iqa_logit_scale,
        device=device,
    )
    write_report(args.log_path, report)


if __name__ == "__main__":
    main()
