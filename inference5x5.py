import argparse
import json
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


def load_eit(ckpt_path, device, enable_easa=False, easa_hidden_dim=64, easa_init_bias=2.0):
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
    eit = torch.nn.DataParallel(eit)
    ckpt = torch.load(ckpt_path, map_location=device)
    eit.load_state_dict(ckpt)
    eit.eval()
    return eit.to(device)


def load_pipe(sdxl_path, device):
    import torch
    from diffusers import StableDiffusionXLPipeline

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--prompt_json", type=str, default="val_prompt.json")
    parser.add_argument("--output_dir", type=str, default="./results/val_prompt_5x5")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/scale_factor_1.5.pth")
    parser.add_argument("--sdxl_path", type=str, default="/root/shared-nvme/model/stable-diffusion-xl-base-1.0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--enable_easa", action="store_true")
    parser.add_argument("--easa_hidden_dim", type=int, default=64)
    parser.add_argument("--easa_init_bias", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output_dir = Path(args.output_dir)
    prompts = [args.prompt] if args.prompt else load_prompts(args.prompt_json)

    if args.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available in this environment.")

    eit = load_eit(
        args.ckpt_path,
        args.device,
        enable_easa=args.enable_easa,
        easa_hidden_dim=args.easa_hidden_dim,
        easa_init_bias=args.easa_init_bias,
    )
    pipe = load_pipe(args.sdxl_path, args.device)

    total = len(prompts) * len(VALENCES) * len(AROUSALS)
    saved_total = 0
    skipped_total = 0
    print(f"Generating {total} images for {len(prompts)} prompt(s) into {output_dir}")

    for prompt_index, prompt in enumerate(prompts, start=1):
        saved_count, skipped_count = emoticrafter5x5(
            pipe,
            eit,
            prompt,
            output_dir=output_dir,
            device=args.device,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        saved_total += saved_count
        skipped_total += skipped_count
        done = saved_total + skipped_total
        print(
            f"[{prompt_index}/{len(prompts)}] saved={saved_count} skipped={skipped_count} "
            f"overall={done}/{total} prompt={prompt}",
            flush=True,
        )

    print(f"Done. saved={saved_total} skipped={skipped_total} output_dir={output_dir}")
