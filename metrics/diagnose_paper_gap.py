import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.clip_score import CLIPScore  # noqa: E402


DEFAULT_IMAGE_DIR = PROJECT_ROOT / "results" / "val_prompt_5x5"
DEFAULT_CLIP_MODEL_PATH = Path("/root/shared-nvme/model/clip-vit-large-patch14")
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
    pyiqa_score: Optional[float] = None


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
    metric = CLIPScore(model_path=model_path, device=device)
    for batch_index, batch in enumerate(chunked(records, batch_size), start=1):
        image_paths = [record.path for record in batch]
        prompts = [record.prompt for record in batch]
        scores = metric.score_pairs(image_paths, prompts)
        for record, score in zip(batch, scores):
            record.clip_score = float(score)
        done = min(batch_index * batch_size, len(records))
        print(f"CLIPScore batch {batch_index}: processed {done}/{len(records)}", flush=True)


def try_import_pyiqa():
    try:
        import pyiqa

        return pyiqa, None
    except ImportError as error:
        return None, error


def image_to_tensor(image_path: Path, device: str) -> torch.Tensor:
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def scalar_from_metric_output(output) -> float:
    if isinstance(output, torch.Tensor):
        return float(output.detach().cpu().flatten().mean().item())
    return float(np.asarray(output, dtype=np.float32).mean())


def compute_pyiqa_scores(
    records: Sequence[ImageRecord],
    metric_name: str,
    device: str,
    input_mode: str,
    strict: bool,
):
    pyiqa, import_error = try_import_pyiqa()
    if pyiqa is None:
        message = (
            "Official CLIP-IQA skipped: pyiqa is not installed in this environment. "
            "Install pyiqa or run this script in the paper's evaluation environment."
        )
        if strict:
            raise RuntimeError(message) from import_error
        print(message, flush=True)
        return message

    metric = pyiqa.create_metric(metric_name, device=device)
    with torch.no_grad():
        for index, record in enumerate(records, start=1):
            if input_mode == "path":
                output = metric(str(record.path))
            elif input_mode == "tensor":
                output = metric(image_to_tensor(record.path, device))
            else:
                raise ValueError("input_mode must be 'path' or 'tensor'")

            record.pyiqa_score = scalar_from_metric_output(output)
            if index % 100 == 0 or index == len(records):
                print(f"{metric_name} batch: processed {index}/{len(records)}", flush=True)

    return None


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
    device: str,
    pyiqa_metric: str,
    pyiqa_skip_reason: Optional[str],
) -> str:
    lines = [
        "EmotiCrafter paper-gap diagnostics",
        f"num_images: {len(records)}",
        f"input_dir: {image_dir}",
        f"clip_model_path: {clip_model_path}",
        f"device: {device}",
        "",
    ]

    if collect_scores(records, "clip_score"):
        add_group_summary(lines, records, "clip_score", "CLIPScore")
        lines.append("")

    if collect_scores(records, "pyiqa_score"):
        add_group_summary(lines, records, "pyiqa_score", f"Official CLIP-IQA ({pyiqa_metric})")
    elif pyiqa_skip_reason:
        lines.append(pyiqa_skip_reason)

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
    parser.add_argument("--skip_pyiqa", action="store_true")
    parser.add_argument("--pyiqa_metric", type=str, default="clipiqa")
    parser.add_argument("--pyiqa_input", choices=["path", "tensor"], default="path")
    parser.add_argument("--strict_pyiqa", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch_size must be greater than 0")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    records = load_records(args.image_dir, limit=args.limit)

    if not args.skip_clipscore:
        compute_clip_scores(
            records=records,
            model_path=args.clip_model_path,
            device=device,
            batch_size=args.batch_size,
        )

    pyiqa_skip_reason = None
    if not args.skip_pyiqa:
        pyiqa_skip_reason = compute_pyiqa_scores(
            records=records,
            metric_name=args.pyiqa_metric,
            device=device,
            input_mode=args.pyiqa_input,
            strict=args.strict_pyiqa,
        )

    report = build_report(
        records=records,
        image_dir=args.image_dir,
        clip_model_path=args.clip_model_path,
        device=device,
        pyiqa_metric=args.pyiqa_metric,
        pyiqa_skip_reason=pyiqa_skip_reason,
    )
    write_report(args.log_path, report)


if __name__ == "__main__":
    main()
