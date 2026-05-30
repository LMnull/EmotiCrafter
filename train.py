import csv
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config
import wandb
import argparse
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
from model import EmotionInjectionTransformer
import numpy as np

def get_density(alist, vlist):
    import numpy as np
    from sklearn.neighbors import KernelDensity
    data = np.vstack([alist.T[0], vlist.T[0]]).T  
    kde = KernelDensity(kernel='gaussian', bandwidth="silverman")
    kde.fit(data)
    log_density = kde.score_samples(data)
    density = np.exp(log_density)
    return density


class EmotionDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'neutral_prompt_feature': torch.as_tensor(item['neutral_prompt_feature'], dtype=torch.float32),
            'arousal': torch.as_tensor(item['arousal'], dtype=torch.float32).reshape(1),
            'valence': torch.as_tensor(item['valence'], dtype=torch.float32).reshape(1),
            'emotional_prompt_feature': torch.as_tensor(item['emotional_prompt_feature'], dtype=torch.float32),
            'density': torch.as_tensor([item['density']], dtype=torch.float32),
        }

def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model

def normalize_state_dict(state_dict):
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint must be a state_dict or contain a 'state_dict' entry.")
    if any(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict

def load_model_state(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    unwrap_model(model).load_state_dict(normalize_state_dict(checkpoint))

def validate_input_file(path, arg_name, expected_description):
    if os.path.isdir(path):
        raise SystemExit(
            f"{arg_name} expects {expected_description}, but got a directory: {path}\n"
            "The SDXL base model directory is used by preprocess.py/inference*.py, not train.py. "
            "For training data, use --data_cache_path ./data/data-cache.pt after running preprocess.py. "
            "For resuming EIT training, use --load_model checkpoints/best_model.pth; "
            "otherwise omit --load_model."
        )
    if not os.path.isfile(path):
        raise SystemExit(f"{arg_name} does not exist or is not a file: {path}")

def batch_to_device(batch, device, non_blocking=False):
    return {
        key: value.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
        for key, value in batch.items()
    }

def train(model, train_loader, optimizer, criterion, device,alpha=1.0,enable_density = False):
    model.train()
    total_loss = 0
    non_blocking = device.type == "cuda"
    
    for batch in tqdm(train_loader, desc="Training"):
        batch = batch_to_device(batch, device, non_blocking=non_blocking)
        neutral_prompt_feature = batch['neutral_prompt_feature']
        arousal = batch['arousal']
        valence = batch['valence']
        emotional_prompt_feature = batch['emotional_prompt_feature']
        density = batch['density']
        optimizer.zero_grad(set_to_none=True)
        predicted_emotional_prompt_feature = model(inputs_embeds=neutral_prompt_feature, arousal=arousal, valence=valence)[0]
        if enable_density:
            loss = criterion(
                    predicted_emotional_prompt_feature*1/density.unsqueeze(-1), 
                    (1/density.unsqueeze(-1))*((emotional_prompt_feature-neutral_prompt_feature)*alpha+neutral_prompt_feature ))
        else:
            loss = criterion(
                    predicted_emotional_prompt_feature, (emotional_prompt_feature-neutral_prompt_feature)*alpha+neutral_prompt_feature )
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device,alpha=1.0,enable_density = False):
    model.eval()
    total_loss = 0
    total_loss_weight = 0
    non_blocking = device.type == "cuda"
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = batch_to_device(batch, device, non_blocking=non_blocking)
            neutral_prompt_feature = batch['neutral_prompt_feature']
            arousal = batch['arousal']
            valence = batch['valence']
            emotional_prompt_feature = batch['emotional_prompt_feature']
            density = batch['density']
            
            predicted_emotional_prompt_feature = model(inputs_embeds=neutral_prompt_feature, arousal=arousal, valence=valence)[0]
            if enable_density:
                loss_weight = criterion(
                predicted_emotional_prompt_feature*1/density.unsqueeze(-1), 
                (1/density.unsqueeze(-1))*((emotional_prompt_feature-neutral_prompt_feature)*alpha + neutral_prompt_feature))
                total_loss_weight+=loss_weight.item()
            loss = criterion(predicted_emotional_prompt_feature, (emotional_prompt_feature-neutral_prompt_feature)*alpha+neutral_prompt_feature )

            total_loss += loss.item()
    return total_loss / len(val_loader), total_loss_weight/len(val_loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument(
        '--load_model',
        type=str,
        default=None,
        help='Optional EIT checkpoint .pth file for resuming training; do not pass the SDXL model directory.',
    )
    parser.add_argument('--device_cuda', type=str, default="0,1")
    parser.add_argument('--scale_factor', type=float, default=1.0)
    parser.add_argument('--wandb_name', type=str, default="your experiment name")
    parser.add_argument('--enable_density',type=bool,default=False)
    parser.add_argument(
        '--data_cache_path',
        type=str,
        default="./data/data-cache.pt",
        help='Preprocessed training cache produced by preprocess.py.',
    )
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()
    
    if args.device_cuda and args.device_cuda.lower() not in {"all", "auto"}:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_cuda

    use_wandb = False
    if use_wandb:
        wandb.init(project="emoticrafter", name=f"{args.wandb_name}", config=vars(args))
    print(f"args\n{args}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count() if device.type == "cuda" else 0
    if n_gpu > 1:
        print(f"Using DataParallel on {n_gpu} visible CUDA devices.")
    else:
        print(f"Using device: {device}")
    
    
    validate_input_file(args.data_cache_path, "--data_cache_path", "a preprocessed .pt data cache file")
    data = torch.load(args.data_cache_path, map_location="cpu")
    
    alist,vlist = np.array([item['arousal'] for item in data]),np.array([item['valence'] for item in data])
    h = 1
    den_list = get_density(alist,vlist)
    for index in range(len(data)):
        a,v = data[index]['arousal'],data[index]['valence']
        data[index]
        data[index]['density'] = den_list[index]



    train_data, val_data = train_test_split(data, test_size=0.01, random_state=42)
    train_dataset = EmotionDataset(train_data)
    val_dataset = EmotionDataset(val_data)
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)
    
    alpha = args.scale_factor
    
    
    config = GPT2Config.from_pretrained('./config')
    model = EmotionInjectionTransformer(config,final_out_type="Linear+LN").to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(n_gpu)))
    if args.load_model:
        validate_input_file(args.load_model, "--load_model", "an EIT .pth checkpoint file")
        load_model_state(model, args.load_model, device)
    model.to(device)
    

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.save_dir, 'best_model.pth')
    best_model_path_weight = os.path.join(args.save_dir, 'best_model_weight.pth')
    if args.enable_density:
        best_val_loss_weight = float('inf')
        
    print("Preparation Done!")
    
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device,alpha,enable_density=args.enable_density)
        val_loss,val_loss_weight = evaluate(model, val_loader, criterion, device,alpha, args.enable_density)
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_loss_weight':val_loss_weight
            })
        
        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val loss weight: {val_loss_weight:.4f}")
        
        # Save model
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        if (epoch + 1) % 50 == 0:
            torch.save(unwrap_model(model).state_dict(), os.path.join(args.save_dir, f'model_epoch_{epoch+1}.pth'))
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(unwrap_model(model).state_dict(), best_model_path)
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        if  args.enable_density:
            if val_loss_weight < best_val_loss_weight:
                best_val_loss_weight = val_loss_weight
                torch.save(unwrap_model(model).state_dict(), best_model_path_weight)
                print(f"New best model saved with weighted validation loss: {best_val_loss_weight:.4f}")
    if use_wandb:
        wandb.finish()

if __name__ == '__main__':
    main()