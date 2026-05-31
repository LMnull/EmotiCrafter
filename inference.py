from transformers import GPT2Config
from model import  EmotionInjectionTransformer
from diffusers import StableDiffusionXLPipeline
import torch
import argparse


def normalize_state_dict(state_dict):
    if isinstance(state_dict, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in state_dict and isinstance(state_dict[key], dict):
                state_dict = state_dict[key]
                break
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint must be a state_dict or contain a state_dict-like entry.")
    if any(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def load_eit(ckpt_path, device):
    config = GPT2Config.from_pretrained('./config')
    eit = EmotionInjectionTransformer(config, final_out_type="Linear+LN").to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    eit.load_state_dict(normalize_state_dict(ckpt), strict=True)
    eit.eval()
    return eit.to(device)


def emoticrafter(pipe,eit, prompt,a = 0, v = 0, device = "cuda", seed = 42 ):
    (   prompt_embeds_ori, 
        negative_prompt_embeds,
        pooled_prompt_embeds_ori, 
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=[prompt],
        prompt_2=[prompt],
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
        negative_prompt=None,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
    )
    resolution= int(1024)
    out = eit(inputs_embeds = prompt_embeds_ori.to(torch.float32),arousal=torch.FloatTensor([[a]]).to(device),valence=torch.FloatTensor([[v]]).to(device))
    image =pipe(
        prompt_embeds = out[0].to(torch.float16),
        pooled_prompt_embeds =pooled_prompt_embeds_ori,
        guidance_scale=7.5,
        num_inference_steps=25,
        height=resolution,
        width=resolution,
        generator = torch.manual_seed(seed)
    ).images[0]
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type = str)
    parser.add_argument('--arousal', type = float)
    parser.add_argument('--valence', type = float)
    parser.add_argument('--ckpt_path', type = str , default='./checkpoints/scale_factor_1.5.pth')
    parser.add_argument('--sdxl_path', type = str, default='/root/shared-nvme/model/stable-diffusion-xl-base-1.0')
    parser.add_argument('--seed', type = int, default = 0)
    args = parser.parse_args()
    
    prompt, arousal, valence = args.prompt, args.arousal, args.valence
    
    device = 'cuda'
    ckpt_path = args.ckpt_path 
    sdxl_path = args.sdxl_path

    eit = load_eit(ckpt_path, device)
    
    pipe = StableDiffusionXLPipeline.from_pretrained(sdxl_path, torch_dtype=torch.float16, 
                                                    use_safetensors=True, variant="fp16")
    pipe.to(device)
    
    save_path = f"./results/{prompt}_v{valence:.1f}_a{arousal:.1f}.png"
    emoticrafter(pipe,eit,prompt , a=arousal, v=valence, seed = args.seed).save(save_path)