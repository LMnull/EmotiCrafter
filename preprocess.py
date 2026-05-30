import csv
import torch
import argparse
from diffusers import StableDiffusionXLPipeline
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sdxl_path', type=str, required=True)
    parser.add_argument('--csv_path', type=str, default='./data/prompt_mapping.csv')
    parser.add_argument('--data_cache_path', type=str, default="./data/data-cache.pt")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Prompt batch size for text-encoder preprocessing. 128 is a good RTX 4090 24GB starting point.',
    )
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--disable_tf32', action='store_true')
    return parser.parse_args()


def configure_cuda(enable_tf32=True):
    if not torch.cuda.is_available():
        return
    if enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')


def batched(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield start, items[start:start + batch_size]


def encode_prompt_batch(pipe, prompts, device):
    with torch.inference_mode():
        prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
            prompt=prompts,
            prompt_2=prompts,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=None,
            negative_prompt_2=None,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            pooled_prompt_embeds=None,
            negative_pooled_prompt_embeds=None,
        )
    return prompt_embeds.detach().cpu().to(torch.float16), pooled_prompt_embeds.detach().cpu().to(torch.float16)


def encode_prompt_pair_batch(pipe, neutral_prompts, emotional_prompts, device):
    row_count = len(neutral_prompts)
    prompt_features, pooled_prompt_features = encode_prompt_batch(
        pipe,
        neutral_prompts + emotional_prompts,
        device,
    )
    return (
        prompt_features[:row_count],
        pooled_prompt_features[:row_count],
        prompt_features[row_count:],
        pooled_prompt_features[row_count:],
    )


if __name__ == "__main__":
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch_size must be greater than 0")
    configure_cuda(enable_tf32=not args.disable_tf32)
    
    with open(args.csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    device = args.device
    sdxl_path = args.sdxl_path
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available in this environment.")

    dtype = torch.float16 if device.startswith('cuda') else torch.float32
    pipe = StableDiffusionXLPipeline.from_pretrained(
        sdxl_path,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if device.startswith('cuda') else None,
    )
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
        
    res_list = []
    progress = tqdm(total=len(data), desc=f"Encoding prompts (batch={args.batch_size})")
    for _, batch in batched(data, args.batch_size):
        neutral_prompts = [item['Neutral_Prompt'] for item in batch]
        emotional_prompts = [item['Emotional_Prompt'] for item in batch]

        (
            neutral_prompt_features,
            neutral_pooled_prompt_features,
            emotional_prompt_features,
            emotional_pooled_prompt_features,
        ) = encode_prompt_pair_batch(pipe, neutral_prompts, emotional_prompts, device)

        for index, item in enumerate(batch):
            res_list.append({
                'neutral_prompt_feature': neutral_prompt_features[index],
                'neutral_pooled_prompt_feature': neutral_pooled_prompt_features[index],
                'arousal': torch.tensor([float(item['Arousal'])], dtype=torch.float),
                'valence': torch.tensor([float(item['Valence'])], dtype=torch.float),
                'emotional_prompt_feature': emotional_prompt_features[index],
                'emotional_pooled_prompt_feature': emotional_pooled_prompt_features[index],
            })
        progress.update(len(batch))
    progress.close()
    torch.save(res_list, args.data_cache_path)
