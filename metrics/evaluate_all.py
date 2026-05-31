import argparse
import json
import multiprocessing
import queue
import re
import sys
from datetime import datetime, timezone
from time import perf_counter
from pathlib import Path
from typing import Dict, List, Sequence, Union

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
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
IMAGE_NAME_RE = re.compile(r"^(?P<prompt>.+)_v(?P<valence>-?\d+(?:\.\d+)?)_a(?P<arousal>-?\d+(?:\.\d+)?)$")


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


def parse_image_record(image_path: Path) -> Dict[str, Union[str, float]]:
    match = IMAGE_NAME_RE.match(image_path.stem)
    if match is None:
        raise ValueError(f"Cannot parse prompt/valence/arousal from image name: {image_path.name}")
    return {
        "path": str(image_path),
        "prompt": match.group("prompt"),
        "target_valence": float(match.group("valence")),
        "target_arousal": float(match.group("arousal")),
    }


def load_image_records(image_dir: Union[str, Path], limit: int = None) -> List[Dict[str, Union[str, float]]]:
    image_dir = Path(image_dir)
    records = [
        parse_image_record(image_path)
        for image_path in sorted(image_dir.iterdir())
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError(f"No valid images found in {image_dir}")
    return records


def chunked(items: Sequence, batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index:index + batch_size]


def split_records(records: Sequence[Dict[str, Union[str, float]]], devices: Sequence[str]):
    chunks = [[] for _ in devices]
    for index, record in enumerate(records):
        chunks[index % len(devices)].append(record)
    return chunks


def resolve_device(device: str) -> str:
    if device:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def normalize_device_name(device: str) -> str:
    device = device.strip()
    if device.isdigit():
        return f"cuda:{device}"
    if device == "cuda":
        return "cuda:0"
    return device


def resolve_devices(device: str, devices: str) -> List[str]:
    if devices:
        resolved = [normalize_device_name(item) for item in devices.split(",") if item.strip()]
    elif device:
        resolved = [normalize_device_name(device)]
    else:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            resolved = [f"cuda:{index}" for index in range(min(2, torch.cuda.device_count()))]
        else:
            resolved = [resolve_device(None)]

    if not resolved:
        raise ValueError("At least one device must be specified.")
    return resolved


def set_cuda_device(device: str):
    if not device.startswith("cuda"):
        return
    import torch

    cuda_device = torch.device(device)
    if cuda_device.index is not None:
        torch.cuda.set_device(cuda_device.index)


def evaluate_record_chunk(worker_id: int, device: str, records: Sequence[Dict[str, Union[str, float]]], config: Dict):
    import gc
    import torch

    from metrics.clip_iqa import CLIPIQA
    from metrics.clip_score import CLIPScore
    from metrics.va_evaluate import va_predictor

    set_cuda_device(device)
    image_paths = [record["path"] for record in records]
    prompts = [record["prompt"] for record in records]
    target_valence = np.asarray([record["target_valence"] for record in records], dtype=np.float64)
    target_arousal = np.asarray([record["target_arousal"] for record in records], dtype=np.float64)

    print(f"[worker {worker_id} {device}] loading VA predictor", flush=True)
    va_metric = va_predictor(
        model_path=config["va_model_path"],
        device=device,
        arousal_ckpt=config["arousal_ckpt"],
        valence_ckpt=config["valence_ckpt"],
    )
    pred_valences: List[float] = []
    pred_arousals: List[float] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(chunked(image_paths, config["batch_size"]), start=1):
            pred_valence, pred_arousal = va_metric(batch)
            pred_valences.extend(pred_valence.detach().cpu().numpy().astype(np.float64).tolist())
            pred_arousals.extend(pred_arousal.detach().cpu().numpy().astype(np.float64).tolist())
            print(
                f"[worker {worker_id} {device}] VA batch {batch_index}: "
                f"{len(pred_valences)}/{len(records)}",
                flush=True,
            )
    del va_metric
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()

    print(f"[worker {worker_id} {device}] loading CLIPScore", flush=True)
    clip_score_metric = CLIPScore(
        model_path=config["clip_score_model_path"],
        device=device,
        text_prefix=config["clip_score_text_prefix"],
        score_weight=config["clip_score_weight"],
        output_scale=config["clip_score_output_scale"],
    )
    clip_scores: List[float] = []
    for batch_index, batch_records in enumerate(chunked(list(zip(image_paths, prompts)), config["batch_size"]), start=1):
        batch_paths = [path for path, _ in batch_records]
        batch_prompts = [prompt for _, prompt in batch_records]
        scores = clip_score_metric.score_pairs(batch_paths, batch_prompts)
        clip_scores.extend(float(score) for score in scores)
        print(
            f"[worker {worker_id} {device}] CLIPScore batch {batch_index}: "
            f"{len(clip_scores)}/{len(records)}",
            flush=True,
        )
    del clip_score_metric
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    gc.collect()

    print(f"[worker {worker_id} {device}] loading CLIP-IQA", flush=True)
    clip_iqa_metric = CLIPIQA(
        model_path=config["clip_iqa_model_path"],
        device=device,
        prompt_pair=(config["clip_iqa_positive_prompt"], config["clip_iqa_negative_prompt"]),
        logit_scale=config["clip_iqa_logit_scale"],
        backend=config["clip_iqa_backend"],
    )
    clip_iqa_scores: List[float] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(chunked(image_paths, config["batch_size"]), start=1):
            scores = clip_iqa_metric.compute_quality_scores(batch)
            clip_iqa_scores.extend(float(score) for score in scores)
            print(
                f"[worker {worker_id} {device}] CLIP-IQA batch {batch_index}: "
                f"{len(clip_iqa_scores)}/{len(records)}",
                flush=True,
            )

    pred_valence_array = np.asarray(pred_valences, dtype=np.float64)
    pred_arousal_array = np.asarray(pred_arousals, dtype=np.float64)
    return {
        "target_valence": target_valence.tolist(),
        "target_arousal": target_arousal.tolist(),
        "pred_valence": pred_valence_array.tolist(),
        "pred_arousal": pred_arousal_array.tolist(),
        "valence_abs_error": np.abs(pred_valence_array - target_valence).tolist(),
        "arousal_abs_error": np.abs(pred_arousal_array - target_arousal).tolist(),
        "clip_score": clip_scores,
        "clip_iqa": clip_iqa_scores,
    }


def worker_entry(worker_id: int, device: str, records, config: Dict, result_queue):
    try:
        result_queue.put({
            "worker_id": worker_id,
            "device": device,
            "result": evaluate_record_chunk(worker_id, device, records, config),
        })
    except Exception as exc:
        result_queue.put({"worker_id": worker_id, "device": device, "error": repr(exc)})
        raise


def merge_worker_results(worker_results: Sequence[Dict]):
    merged = {
        "target_valence": [],
        "target_arousal": [],
        "pred_valence": [],
        "pred_arousal": [],
        "valence_abs_error": [],
        "arousal_abs_error": [],
        "clip_score": [],
        "clip_iqa": [],
    }
    for worker_result in worker_results:
        for key in merged:
            merged[key].extend(worker_result[key])

    va_results = {
        key: np.asarray(merged[key], dtype=np.float64)
        for key in (
            "target_valence",
            "target_arousal",
            "pred_valence",
            "pred_arousal",
            "valence_abs_error",
            "arousal_abs_error",
        )
    }
    return va_results, np.asarray(merged["clip_score"], dtype=np.float64), np.asarray(merged["clip_iqa"], dtype=np.float64)


def run_multi_device(records: Sequence[Dict[str, Union[str, float]]], devices: Sequence[str], config: Dict):
    chunks = split_records(records, devices)
    active_jobs = [(device, chunk) for device, chunk in zip(devices, chunks) if chunk]

    if len(active_jobs) == 1:
        result = evaluate_record_chunk(0, active_jobs[0][0], active_jobs[0][1], config)
        return merge_worker_results([result])

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = []
    for worker_id, (device, chunk) in enumerate(active_jobs):
        process = context.Process(target=worker_entry, args=(worker_id, device, chunk, config, result_queue))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    queue_results = []
    for _ in processes:
        try:
            queue_results.append(result_queue.get(timeout=1))
        except queue.Empty:
            pass

    errors = [result for result in queue_results if "error" in result]
    failed_processes = [process.pid for process in processes if process.exitcode != 0]
    if len(queue_results) != len(processes):
        errors.append({"error": f"Collected {len(queue_results)} result(s) from {len(processes)} worker(s)."})
    if errors or failed_processes:
        raise RuntimeError(f"Multi-device evaluation failed. errors={errors}, failed_processes={failed_processes}")

    queue_results.sort(key=lambda item: item["worker_id"])
    return merge_worker_results([item["result"] for item in queue_results])


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VA, CLIPScore, and CLIP-IQA in one run.")
    parser.add_argument("--image_dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--log_path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--devices", type=str, default=None, help="Comma-separated devices, e.g. cuda:0,cuda:1")
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
    if args.batch_size < 1:
        raise ValueError("batch_size must be greater than 0")

    devices = resolve_devices(args.device, args.devices)
    used_device = ",".join(devices)
    started_at = datetime.now(timezone.utc).astimezone()
    start_time = started_at.isoformat(timespec="seconds")
    start_tick = perf_counter()

    records = load_image_records(args.image_dir, limit=args.limit)
    config = {
        "batch_size": args.batch_size,
        "va_model_path": args.va_model_path,
        "arousal_ckpt": args.arousal_ckpt,
        "valence_ckpt": args.valence_ckpt,
        "clip_score_model_path": args.clip_score_model_path,
        "clip_score_text_prefix": args.clip_score_text_prefix,
        "clip_score_weight": args.clip_score_weight,
        "clip_score_output_scale": args.clip_score_output_scale,
        "clip_iqa_model_path": args.clip_iqa_model_path,
        "clip_iqa_positive_prompt": args.clip_iqa_positive_prompt,
        "clip_iqa_negative_prompt": args.clip_iqa_negative_prompt,
        "clip_iqa_logit_scale": args.clip_iqa_logit_scale,
        "clip_iqa_backend": args.clip_iqa_backend,
    }
    print(f"Running all metrics on {len(records)} image(s) with devices={used_device}", flush=True)
    va_results, clip_score_values, clip_iqa_values = run_multi_device(records, devices, config)

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
