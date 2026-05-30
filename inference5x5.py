import argparse
import json
import multiprocessing as mp
import queue
import traceback
from pathlib import Path
from typing import Dict, List, Sequence


VALENCES = [-3, -1.5, 0, 1.5, 3]
AROUSALS = [-3, -1.5, 0, 1.5, 3]


def format_va_value(value):
    return f"{value:g}"


def safe_prompt_filename(prompt):
    return "".join("_" if char in '/\\\0' else char for char in prompt.strip())


def image_path(output_dir, prompt, valence, arousal):
    safe_prompt = safe_prompt_filename(prompt)
    v_text = format_va_value(valence)
    a_text = format_va_value(arousal)
    return output_dir / f"{safe_prompt}_v{v_text}_a{a_text}.png"


def load_prompts(prompt_json):
    with open(prompt_json, "r", encoding="utf-8") as file:
        data = json.load(file)

    prompts = data["prompts"] if isinstance(data, dict) else data
    if not isinstance(prompts, list) or not all(isinstance(prompt, str) for prompt in prompts):
        raise ValueError(f"{prompt_json} must contain a prompt list or a dict with a 'prompts' list")
    return prompts


def parse_device_ids(device_cuda: str) -> List[int]:
    if device_cuda.lower() in ("", "none", "cpu"):
        return []
    return [int(device_id.strip()) for device_id in device_cuda.split(",") if device_id.strip()]


def normalize_state_dict_for_model(state_dict, model_is_parallel):
    has_module_prefix = any(key.startswith("module.") for key in state_dict.keys())
    if model_is_parallel and not has_module_prefix:
        return {f"module.{key}": value for key, value in state_dict.items()}
    if not model_is_parallel and has_module_prefix:
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def load_eit(
    ckpt_path,
    device,
    enable_easa=False,
    easa_hidden_dim=64,
    easa_init_bias=2.0,
    data_parallel=False,
):
    import torch
    from transformers import GPT2Config

    from model import EmotionInjectionTransformer

    config = GPT2Config.from_pretrained("./config")
    eit = EmotionInjectionTransformer(
        config,
        final_out_type="Linear+LN",
        use_easa=enable_easa,
        easa_hidden_dim=easa_hidden_dim,
        easa_init_bias=easa_init_bias,
    ).to(device)
    if data_parallel:
        eit = torch.nn.DataParallel(eit)

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    state_dict = normalize_state_dict_for_model(state_dict, data_parallel)
    eit.load_state_dict(state_dict)
    eit.eval()
    return eit.to(device)


def load_pipe(sdxl_path, device):
    import torch
    from diffusers import StableDiffusionXLPipeline

    device_text = str(device)
    torch_dtype = torch.float16 if device_text.startswith("cuda") else torch.float32
    pipe = StableDiffusionXLPipeline.from_pretrained(
        sdxl_path,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        variant="fp16" if device_text.startswith("cuda") else None,
    )
    return pipe.to(device)


def build_generation_tasks(prompts: Sequence[str], output_dir: Path, overwrite: bool):
    tasks = []
    skipped_count = 0
    for prompt_index, prompt in enumerate(prompts, start=1):
        for arousal in AROUSALS:
            for valence in VALENCES:
                save_path = image_path(output_dir, prompt, valence, arousal)
                if save_path.exists() and not overwrite:
                    skipped_count += 1
                    continue
                tasks.append(
                    {
                        "prompt_index": prompt_index,
                        "prompt": prompt,
                        "arousal": arousal,
                        "valence": valence,
                        "save_path": str(save_path),
                    }
                )
    return tasks, skipped_count


def split_tasks_round_robin(tasks: Sequence[Dict], num_shards: int) -> List[List[Dict]]:
    shards = [[] for _ in range(num_shards)]
    for task_index, task in enumerate(tasks):
        shards[task_index % num_shards].append(task)
    return shards


def run_generation_tasks(tasks, pipe, eit, device, seed, worker_name):
    import torch

    from inference import emoticrafter

    saved_count = 0
    with torch.no_grad():
        for task in tasks:
            save_path = Path(task["save_path"])
            image = emoticrafter(
                pipe,
                eit,
                task["prompt"],
                task["arousal"],
                task["valence"],
                device=device,
                seed=seed,
            )
            image.save(save_path)
            saved_count += 1
            print(
                f"[{worker_name}] saved={saved_count}/{len(tasks)} "
                f"prompt_index={task['prompt_index']} "
                f"v={task['valence']} a={task['arousal']} path={save_path}",
                flush=True,
            )
    return saved_count


def worker_main(worker_index: int, device_id: int, tasks: List[Dict], args_dict: Dict, result_queue):
    try:
        import torch

        device = f"cuda:{device_id}"
        torch.cuda.set_device(device_id)
        worker_name = f"gpu{device_id}"
        print(f"[{worker_name}] loading models for {len(tasks)} task(s)", flush=True)

        eit = load_eit(
            args_dict["ckpt_path"],
            device,
            enable_easa=args_dict["enable_easa"],
            easa_hidden_dim=args_dict["easa_hidden_dim"],
            easa_init_bias=args_dict["easa_init_bias"],
            data_parallel=False,
        )
        pipe = load_pipe(args_dict["sdxl_path"], device)
        saved_count = run_generation_tasks(
            tasks=tasks,
            pipe=pipe,
            eit=eit,
            device=device,
            seed=args_dict["seed"],
            worker_name=worker_name,
        )
        result_queue.put({"worker_index": worker_index, "device": device, "saved": saved_count, "error": None})
    except Exception:
        result_queue.put(
            {
                "worker_index": worker_index,
                "device": f"cuda:{device_id}",
                "saved": 0,
                "error": traceback.format_exc(),
            }
        )


def run_multi_gpu(tasks: Sequence[Dict], device_ids: Sequence[int], args):
    import torch

    available_count = torch.cuda.device_count()
    invalid_ids = [device_id for device_id in device_ids if device_id < 0 or device_id >= available_count]
    if invalid_ids:
        raise ValueError(
            f"Invalid CUDA device ids {invalid_ids}; this machine exposes {available_count} CUDA device(s)."
        )

    shards = split_tasks_round_robin(tasks, len(device_ids))
    args_dict = {
        "ckpt_path": args.ckpt_path,
        "sdxl_path": args.sdxl_path,
        "seed": args.seed,
        "enable_easa": args.enable_easa,
        "easa_hidden_dim": args.easa_hidden_dim,
        "easa_init_bias": args.easa_init_bias,
    }

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes = []
    for worker_index, (device_id, shard) in enumerate(zip(device_ids, shards)):
        process = context.Process(
            target=worker_main,
            args=(worker_index, device_id, shard, args_dict, result_queue),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    results = []
    while len(results) < len(processes):
        try:
            results.append(result_queue.get_nowait())
        except queue.Empty:
            break

    errors = [result for result in results if result["error"]]
    missing_results = len(processes) - len(results)
    failed_processes = [
        process.exitcode
        for process in processes
        if process.exitcode not in (0, None)
    ]
    if missing_results or failed_processes:
        raise SystemExit(
            f"{missing_results} worker result(s) missing; failed worker exit codes: {failed_processes}"
        )
    if errors:
        for result in errors:
            print(f"[{result['device']}] worker failed:\n{result['error']}", flush=True)
        raise SystemExit("At least one GPU worker failed.")

    return sum(result["saved"] for result in results)


def run_single_device(tasks: Sequence[Dict], args):
    import torch

    device = args.device
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available in this environment.")
        if ":" in device:
            torch.cuda.set_device(int(device.split(":", 1)[1]))

    eit = load_eit(
        args.ckpt_path,
        device,
        enable_easa=args.enable_easa,
        easa_hidden_dim=args.easa_hidden_dim,
        easa_init_bias=args.easa_init_bias,
        data_parallel=False,
    )
    pipe = load_pipe(args.sdxl_path, device)
    return run_generation_tasks(
        tasks=tasks,
        pipe=pipe,
        eit=eit,
        device=device,
        seed=args.seed,
        worker_name=str(device),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_json", type=str, default="val_prompt.json")
    parser.add_argument("--output_dir", type=str, default="./results/val_prompt_5x5")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/scale_factor_1.5.pth")
    parser.add_argument("--sdxl_path", type=str, default="/root/shared-nvme/model/stable-diffusion-xl-base-1.0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device_cuda", type=str, default="0,1", help="Comma-separated CUDA ids for multi-GPU mode.")
    parser.add_argument("--single_device", action="store_true", help="Disable multi-GPU sharding.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--enable_easa", action="store_true")
    parser.add_argument("--easa_hidden_dim", type=int, default=64)
    parser.add_argument("--easa_init_bias", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = [args.prompt] if args.prompt else load_prompts(args.prompt_json)
    tasks, skipped_total = build_generation_tasks(prompts, output_dir, args.overwrite)
    total = len(prompts) * len(VALENCES) * len(AROUSALS)

    print(
        f"Generating {total} image target(s) for {len(prompts)} prompt(s) into {output_dir}. "
        f"pending={len(tasks)} skipped={skipped_total}",
        flush=True,
    )

    if not tasks:
        print(f"Done. saved=0 skipped={skipped_total} output_dir={output_dir}")
        raise SystemExit(0)

    device_ids = parse_device_ids(args.device_cuda)
    if not args.single_device and len(device_ids) > 1:
        saved_total = run_multi_gpu(tasks, device_ids, args)
    else:
        saved_total = run_single_device(tasks, args)

    print(f"Done. saved={saved_total} skipped={skipped_total} output_dir={output_dir}")
