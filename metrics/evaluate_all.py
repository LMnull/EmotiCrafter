import argparse
import inspect
import json
import sys
from datetime import datetime, timezone
from time import perf_counter
from pathlib import Path
from typing import Dict, Sequence, Union

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

METRICS_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "results" / "val_prompt_5x5"
DEFAULT_LOG_PATH = PROJECT_ROOT / "log.txt"
DEFAULT_VA_MODEL_PATH = Path("/root/shared-nvme/model/clip-vit-base-patch32")
DEFAULT_CLIP_SCORE_MODEL_PATH = Path("/root/shared-nvme/model/clip-vit-base-patch32")
DEFAULT_CLIP_IQA_MODEL_PATH = Path("/root/shared-nvme/model/RN50.pt")
DEFAULT_AROUSAL_CKPT = METRICS_DIR / "arousal1_CLIP_lr=0.001_loss=MSELoss_sc=test_cuda-1.pth"
DEFAULT_VALENCE_CKPT = METRICS_DIR / "valence1_CLIP_lr=0.0001_loss=MSELoss_sc=test_cuda.pth"


def format_mean_std(values: Sequence[float]) -> str:
    values = np.asarray(values, dtype=np.float64)
    return f"{values.mean():.4f} ± {values.std(ddof=0):.4f}"


def summarize_va(results: Dict[str, np.ndarray]) -> Dict[str, str]:
    return {
        "pred_valence": format_mean_std(results["pred_valence"]),
        "pred_arousal": format_mean_std(results["pred_arousal"]),
        "valence_abs_error": format_mean_std(results["valence_abs_error"]),
        "arousal_abs_error": format_mean_std(results["arousal_abs_error"]),
        "num_images": str(len(results["pred_valence"])),
    }


def summarize_array(values: Union[np.ndarray, Sequence[float]]) -> Dict[str, str]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "score": format_mean_std(values),
        "num_images": str(len(values)),
    }


def args_to_loggable_dict(args: argparse.Namespace, used_device: str) -> Dict[str, str]:
    values = vars(args).copy()
    values["device"] = used_device
    return {key: str(value) for key, value in values.items()}


def format_log_section(
    start_time: str,
    end_time: str,
    elapsed_seconds: float,
    args: argparse.Namespace,
    used_device: str,
    va_summary: Dict[str, str],
    clip_score_summary: Dict[str, str],
    clip_iqa_summary: Dict[str, str],
) -> str:
    params = args_to_loggable_dict(args, used_device)
    lines = [
        "=" * 80,
        f"start_time: {start_time}",
        f"end_time: {end_time}",
        f"elapsed_seconds: {elapsed_seconds:.2f}",
        "parameters:",
        json.dumps(params, ensure_ascii=False, indent=2, sort_keys=True),
        "metrics:",
        f"  VA pred_valence: {va_summary['pred_valence']}",
        f"  VA pred_arousal: {va_summary['pred_arousal']}",
        f"  VA valence_abs_error: {va_summary['valence_abs_error']}",
        f"  VA arousal_abs_error: {va_summary['arousal_abs_error']}",
        f"  VA num_images: {va_summary['num_images']}",
        f"  CLIPScore: {clip_score_summary['score']}",
        f"  CLIPScore num_images: {clip_score_summary['num_images']}",
        f"  CLIP-IQA: {clip_iqa_summary['score']}",
        f"  CLIP-IQA num_images: {clip_iqa_summary['num_images']}",
    ]
    return "\n".join(lines) + "\n"


def write_log(log_path: Union[str, Path], log_text: str, append: bool):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if append and log_path.exists() and log_path.stat().st_size > 0:
        log_text = "\n" + log_text
    mode = "a" if append else "w"
    with log_path.open(mode, encoding="utf-8") as file:
        file.write(log_text)
    print(log_text, end="")


def resolve_device(device: str) -> str:
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VA, CLIPScore, and CLIP-IQA in one run.")
    parser.add_argument("--image_dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--log_path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite_log", action="store_true")

    parser.add_argument("--va_model_path", type=Path, default=DEFAULT_VA_MODEL_PATH)
    parser.add_argument("--arousal_ckpt", type=Path, default=DEFAULT_AROUSAL_CKPT)
    parser.add_argument("--valence_ckpt", type=Path, default=DEFAULT_VALENCE_CKPT)
    parser.add_argument("--clip_score_model_path", type=Path, default=DEFAULT_CLIP_SCORE_MODEL_PATH)
    parser.add_argument("--clip_score_text_prefix", type=str, default="A photo depicts ")
    parser.add_argument("--clip_score_weight", type=float, default=2.5)
    parser.add_argument("--clip_score_output_scale", type=float, default=1.0)
    parser.add_argument("--clip_iqa_model_path", type=Path, default=DEFAULT_CLIP_IQA_MODEL_PATH)
    parser.add_argument("--clip_iqa_positive_prompt", type=str, default="Good photo.")
    parser.add_argument("--clip_iqa_negative_prompt", type=str, default="Bad photo.")
    parser.add_argument("--clip_iqa_logit_scale", choices=["learned", "100"], default="learned")
    parser.add_argument("--clip_iqa_backend", choices=["auto", "transformers", "openai_clip"], default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    from metrics.clip_iqa import evaluate_directory as evaluate_clip_iqa_directory
    from metrics.clip_score import evaluate_directory as evaluate_clip_score_directory
    from metrics.va_evaluate import evaluate_directory as evaluate_va_directory

    clip_iqa_params = inspect.signature(evaluate_clip_iqa_directory).parameters
    required_clip_iqa_params = {"positive_prompt", "negative_prompt", "logit_scale", "backend"}
    if not required_clip_iqa_params.issubset(clip_iqa_params):
        raise RuntimeError(
            "Loaded metrics.clip_iqa.evaluate_directory does not support the official CLIP-IQA "
            "arguments. Please update metrics/clip_iqa.py together with metrics/evaluate_all.py."
        )

    used_device = resolve_device(args.device)
    started_at = datetime.now(timezone.utc).astimezone()
    start_time = started_at.isoformat(timespec="seconds")
    start_tick = perf_counter()

    print("Running VA evaluation...", flush=True)
    va_results = evaluate_va_directory(
        image_dir=args.image_dir,
        model_path=args.va_model_path,
        arousal_ckpt=args.arousal_ckpt,
        valence_ckpt=args.valence_ckpt,
        device=used_device,
        batch_size=args.batch_size,
        limit=args.limit,
    )

    print("Running CLIPScore evaluation...", flush=True)
    clip_score_values, _ = evaluate_clip_score_directory(
        image_dir=args.image_dir,
        model_path=args.clip_score_model_path,
        device=used_device,
        batch_size=args.batch_size,
        limit=args.limit,
        text_prefix=args.clip_score_text_prefix,
        score_weight=args.clip_score_weight,
        output_scale=args.clip_score_output_scale,
    )

    print("Running CLIP-IQA evaluation...", flush=True)
    clip_iqa_values = evaluate_clip_iqa_directory(
        image_dir=args.image_dir,
        model_path=args.clip_iqa_model_path,
        device=used_device,
        batch_size=args.batch_size,
        limit=args.limit,
        positive_prompt=args.clip_iqa_positive_prompt,
        negative_prompt=args.clip_iqa_negative_prompt,
        logit_scale=args.clip_iqa_logit_scale,
        backend=args.clip_iqa_backend,
    )

    ended_at = datetime.now(timezone.utc).astimezone()
    log_text = format_log_section(
        start_time=start_time,
        end_time=ended_at.isoformat(timespec="seconds"),
        elapsed_seconds=perf_counter() - start_tick,
        args=args,
        used_device=used_device,
        va_summary=summarize_va(va_results),
        clip_score_summary=summarize_array(clip_score_values),
        clip_iqa_summary=summarize_array(clip_iqa_values),
    )
    write_log(args.log_path, log_text, append=not args.overwrite_log)


if __name__ == "__main__":
    main()
