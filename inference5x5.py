import argparse
import json
import multiprocessing
import queue
from pathlib import Path


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


def normalize_state_dict(state_dict):
    if isinstance(state_dict, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint must be a state_dict or contain a 'state_dict' entry.")
    if any(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def set_cuda_device(device):
    if not device.startswith("cuda"):
        return

    import torch

    cuda_device = torch.device(device)
    if cuda_device.index is not None:
        torch.cuda.set_device(cuda_device.index)


def load_eit(ckpt_path, device, use_data_parallel=False):
    import torch
    from transformers import GPT2Config

    from model import EmotionInjectionTransformer

    set_cuda_device(device)
    config = GPT2Config.from_pretrained("./config")
    eit = EmotionInjectionTransformer(config, final_out_type="Linear+LN").to(device)
    if use_data_parallel and device == "cuda" and torch.cuda.device_count() > 1:
        eit = torch.nn.DataParallel(eit)
    ckpt = torch.load(ckpt_path, map_location=device)
    target = eit.module if isinstance(eit, torch.nn.DataParallel) else eit
    target.load_state_dict(normalize_state_dict(ckpt), strict=True)
    eit.eval()
    return eit.to(device)


def load_pipe(sdxl_path, device):
    import torch
    from diffusers import StableDiffusionXLPipeline

    set_cuda_device(device)
    torch_dtype = torch.float16 if device.startswith("cuda") else torch.float32
    pipe = StableDiffusionXLPipeline.from_pretrained(
        sdxl_path,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        variant="fp16" if device.startswith("cuda") else None,
    )
    return pipe.to(device)


def emoticrafter5x5(
    pipe,
    eit,
    prompt,
    output_dir,
    device="cuda",
    seed=42,
    overwrite=False,
):
    import torch

    from inference import emoticrafter

    saved_count = 0
    skipped_count = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for arousal in AROUSALS:
            for valence in VALENCES:
                save_path = image_path(output_dir, prompt, valence, arousal)
                if save_path.exists() and not overwrite:
                    skipped_count += 1
                    continue

                image = emoticrafter(
                    pipe,
                    eit,
                    prompt,
                    arousal,
                    valence,
                    device=device,
                    seed=seed,
                )
                image.save(save_path)
                saved_count += 1

    return saved_count, skipped_count


def build_generation_tasks(prompts, output_dir, overwrite=False):
    tasks = []
    skipped_count = 0

    output_dir.mkdir(parents=True, exist_ok=True)
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


def split_tasks(tasks, devices):
    chunks = [[] for _ in devices]
    for task_index, task in enumerate(tasks):
        chunks[task_index % len(devices)].append(task)
    return chunks


def normalize_device_name(device):
    device = device.strip()
    if device == "cuda":
        return "cuda:0"
    return device


def resolve_devices(device_arg, devices_arg):
    if devices_arg:
        devices = [normalize_device_name(device) for device in devices_arg.split(",") if device.strip()]
    elif "," in device_arg:
        devices = [normalize_device_name(device) for device in device_arg.split(",") if device.strip()]
    elif device_arg == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available in this environment.")
        devices = [f"cuda:{index}" for index in range(min(2, torch.cuda.device_count()))]
    else:
        devices = [normalize_device_name(device_arg)]

    if not devices:
        raise ValueError("At least one device must be specified.")
    return devices


def validate_devices(devices):
    cuda_devices = [device for device in devices if device.startswith("cuda")]
    if not cuda_devices:
        return

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available in this environment.")

    available_count = torch.cuda.device_count()
    for device in cuda_devices:
        cuda_device = torch.device(device)
        index = 0 if cuda_device.index is None else cuda_device.index
        if index >= available_count:
            raise SystemExit(
                f"{device} was requested, but only {available_count} CUDA device(s) are available."
            )


def generate_task_chunk(worker_id, device, tasks, ckpt_path, sdxl_path, seed):
    import torch

    from inference import emoticrafter

    set_cuda_device(device)
    eit = load_eit(ckpt_path, device)
    pipe = load_pipe(sdxl_path, device)

    saved_count = 0
    with torch.no_grad():
        for task_index, task in enumerate(tasks, start=1):
            save_path = Path(task["save_path"])
            save_path.parent.mkdir(parents=True, exist_ok=True)
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
                f"[worker {worker_id} {device}] saved {task_index}/{len(tasks)} {save_path}",
                flush=True,
            )

    return {"worker_id": worker_id, "device": device, "saved": saved_count}


def worker_entry(worker_id, device, tasks, ckpt_path, sdxl_path, seed, result_queue):
    try:
        result_queue.put(generate_task_chunk(worker_id, device, tasks, ckpt_path, sdxl_path, seed))
    except Exception as exc:
        result_queue.put({"worker_id": worker_id, "device": device, "error": repr(exc)})
        raise


def run_generation_tasks(tasks, devices, ckpt_path, sdxl_path, seed):
    chunks = split_tasks(tasks, devices)

    if len(devices) == 1:
        return generate_task_chunk(0, devices[0], chunks[0], ckpt_path, sdxl_path, seed)["saved"]

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    processes = []
    for worker_id, (device, task_chunk) in enumerate(zip(devices, chunks)):
        if not task_chunk:
            continue
        process = context.Process(
            target=worker_entry,
            args=(worker_id, device, task_chunk, ckpt_path, sdxl_path, seed, result_queue),
        )
        process.start()
        processes.append(process)

    for process in processes:
        process.join()

    results = []
    for _ in processes:
        try:
            results.append(result_queue.get(timeout=1))
        except queue.Empty:
            pass

    errors = [result for result in results if "error" in result]
    failed_processes = [process.pid for process in processes if process.exitcode != 0]
    if len(results) != len(processes):
        errors.append({"error": f"Collected {len(results)} result(s) from {len(processes)} worker(s)."})
    if errors or failed_processes:
        raise RuntimeError(f"Generation failed. errors={errors}, failed_processes={failed_processes}")

    return sum(result["saved"] for result in results)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_json", type=str, default="val_prompt.json")
    parser.add_argument("--output_dir", type=str, default="./results/val_prompt_5x5")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/best_model.pth")
    parser.add_argument("--sdxl_path", type=str, default="/root/shared-nvme/model/stable-diffusion-xl-base-1.0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--devices", type=str, default=None, help="Comma-separated devices, e.g. cuda:0,cuda:1")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output_dir = Path(args.output_dir)
    prompts = [args.prompt] if args.prompt else load_prompts(args.prompt_json)
    devices = resolve_devices(args.device, args.devices)
    validate_devices(devices)

    tasks, skipped_total = build_generation_tasks(prompts, output_dir, overwrite=args.overwrite)
    total = len(prompts) * len(VALENCES) * len(AROUSALS)
    print(
        f"Generating {len(tasks)} image(s), skipped={skipped_total}, total={total}, "
        f"devices={','.join(devices)}, output_dir={output_dir}",
        flush=True,
    )

    saved_total = 0
    if tasks:
        saved_total = run_generation_tasks(
            tasks,
            devices,
            ckpt_path=args.ckpt_path,
            sdxl_path=args.sdxl_path,
            seed=args.seed,
        )

    print(f"Done. saved={saved_total} skipped={skipped_total} output_dir={output_dir}")
