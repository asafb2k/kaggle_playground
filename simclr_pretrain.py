import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
import os
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import datetime
import random

# Import custom modules
from models import resnet18_3d
from transforms import Compose, RandomFlip3D, RandomRotate90_3D, RandomIntensityScale, RandomNoise, RandomCutout3D, Normalize

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

class SimCLR(nn.Module):
    def __init__(self, base_model, out_dim=128):
        super(SimCLR, self).__init__()
        self.encoder = base_model
        # Remove the final classification layer of the encoder if it exists
        # In our resnet18_3d implementation, the fc layer is named 'fc'
        # We want to keep the encoder up to the pooling layer
        
        # Check input dim of fc layer
        dim_mlp = self.encoder.fc.in_features
        
        # Replace fc with identity to get features
        self.encoder.fc = nn.Identity()
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        
        # Concatenate representations
        z = torch.cat([z_i, z_j], dim=0)
        
        # Cosine similarity
        z = F.normalize(z, dim=1)
        similarity_matrix = torch.matmul(z, z.T)
        
        # Create mask to remove self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        
        # Discard self-similarity
        # We can just fill diagonal with -inf
        similarity_matrix.masked_fill_(mask, -float('inf'))
        
        # Positives are (i, i+batch_size) and (i+batch_size, i)
        # We want to maximize similarity between z_i and z_j
        
        # Labels for cross entropy
        # For index i (0 to N-1), the positive is i + N
        # For index i + N (N to 2N-1), the positive is i
        
        labels = torch.cat([
            torch.arange(batch_size, device=z.device) + batch_size,
            torch.arange(batch_size, device=z.device)
        ], dim=0)
        
        loss = F.cross_entropy(similarity_matrix / self.temperature, labels)
        
        return loss

def get_simclr_transform(mean=0.15, std=0.25):
    # SimCLR relies heavily on strong data augmentation
    return Compose([
        RandomFlip3D(p=0.5),
        RandomRotate90_3D(p=0.5),
        RandomIntensityScale(scale_limit=0.2, p=0.5), # Stronger intensity
        RandomNoise(std=0.05, p=0.5), # Stronger noise
        RandomCutout3D(length=40, p=0.5), # Larger cutout
        Normalize(mean=mean, std=std)
    ])

class ForamSimCLRDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        volume = tifffile.imread(file_path)
        volume = volume.reshape(1, *volume.shape)
        volume = volume.astype(np.float32) / volume.max()
        
        # Generate two views
        v1 = volume.copy()
        v2 = volume.copy()
        
        if self.transform:
            v1 = self.transform(v1)
            v2 = self.transform(v2)
            
        return torch.tensor(v1, dtype=torch.float32), torch.tensor(v2, dtype=torch.float32)

def train_simclr(model, dataloader, criterion, optimizer, device, writer, epoch, scaler=None):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Pretrain Epoch {epoch}")
    for batch_idx, (x_i, x_j) in enumerate(pbar):
        x_i = x_i.to(device)
        x_j = x_j.to(device)
        
        optimizer.zero_grad()
        
        if scaler:
            with autocast():
                _, z_i = model(x_i)
                _, z_j = model(x_j)
                loss = criterion(z_i, z_j)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            _, z_i = model(x_i)
            _, z_j = model(x_j)
            loss = criterion(z_i, z_j)
            loss.backward()
            optimizer.step()
            
        running_loss += loss.item()
        pbar.set_postfix({'Loss': loss.item()})
        
    epoch_loss = running_loss / len(dataloader)
    writer.add_scalar('Pretrain/Loss', epoch_loss, epoch)
    
    return epoch_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='simclr_resnet18')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32) # Larger batch size helps SimCLR
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--out_dim', type=int, default=128)
    args = parser.parse_args()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{args.exp_name}_{timestamp}"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Directories
    experiments_directory = Path("experiments")
    experiment_path = experiments_directory / experiment_name
    tensorboard_path = experiment_path / 'tensorboard'
    os.makedirs(experiment_path, exist_ok=True)
    os.makedirs(tensorboard_path, exist_ok=True)
    
    writer = SummaryWriter(log_dir=tensorboard_path)
    
    # Data Loading - Use ALL data (labeled + unlabeled)
    labeled_volume_dir = Path(r'data\volumes\volumes\labelled')
    unlabeled_volume_dir = Path(r'data\volumes\volumes\unlabelled')
    
    all_files = []
    
    # Add labeled files
    prefix_l = "labelled_foram_"
    for file in labeled_volume_dir.iterdir():
        if file.name.startswith(prefix_l):
            all_files.append(file)
            
    # Add unlabeled files
    prefix_u = "foram_"
    for file in unlabeled_volume_dir.iterdir():
        if file.name.startswith(prefix_u):
            all_files.append(file)
            
    print(f"Total files for pretraining: {len(all_files)}")
    
    # Transform
    transform = get_simclr_transform()
    
    # Dataset
    dataset = ForamSimCLRDataset(all_files, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    
    # Model
    base_model = resnet18_3d(num_classes=15) # num_classes doesn't matter here as we remove fc
    model = SimCLR(base_model, out_dim=args.out_dim).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = NTXentLoss(temperature=args.temperature)
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        loss = train_simclr(model, dataloader, criterion, optimizer, device, writer, epoch, scaler=scaler)
        
        scheduler.step()
        writer.add_scalar('Pretrain/LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Loss: {loss:.4f}")
        
        if loss < best_loss:
            best_loss = loss
            torch.save(model.encoder.state_dict(), experiment_path / f'simclr_encoder_best_loss_{loss:.4f}.pth')
            print("Saved best encoder")
            
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
             torch.save(model.encoder.state_dict(), experiment_path / f'simclr_encoder_epoch_{epoch+1}.pth')
            
    writer.close()
    print(f"Pretraining complete. Best loss: {best_loss:.4f}")

if __name__ == '__main__':
    main()
