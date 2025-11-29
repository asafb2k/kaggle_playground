import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
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
from transforms import Compose, RandomFlip3D, RandomRotate90_3D, RandomIntensityScale, RandomNoise, Normalize

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

def get_train_transform(mean=0.15, std=0.25):
    return Compose([
        RandomFlip3D(p=0.5),
        RandomRotate90_3D(p=0.5),
        RandomIntensityScale(scale_limit=0.1, p=0.5),
        RandomNoise(std=0.02, p=0.5),
        Normalize(mean=mean, std=std)
    ])

def get_val_transform(mean=0.15, std=0.25):
    return Compose([
        Normalize(mean=mean, std=std)
    ])

class ForamDataset(Dataset):
    def __init__(self, data_frame, volume_dir, transform=None):
        self.data_frame = data_frame
        self.volume_dir = volume_dir
        self.transform = transform
        
        self.file_paths = {}
        prefix = "labelled_foram_"
        for file in self.volume_dir.iterdir():
            if file.name.startswith(prefix):
                file_id = int(file.name.split('_')[2])
                self.file_paths[file_id] = self.volume_dir / file.name
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        file_id = int(self.data_frame.iloc[idx]['id'].split('_')[1])
        label = self.data_frame.iloc[idx]['label']
        file_path = self.file_paths.get(file_id)
        
        volume = tifffile.imread(file_path)
        volume = volume.reshape(1, *volume.shape)
        volume = volume.astype(np.float32) / volume.max()
        
        if self.transform:
            volume = self.transform(volume)
            
        return torch.tensor(volume, dtype=torch.float32), label

def train_epoch(model, dataloader, criterion, optimizer, device, writer, epoch, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if scaler:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/train', epoch_acc, epoch)
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device, writer, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    # For per-class accuracy tracking
    class_correct = np.zeros(15)
    class_total = np.zeros(15)
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Track per-class accuracy
            for c in range(15):
                class_mask = (labels == c)
                class_correct[c] += (predicted[class_mask] == c).sum().item()
                class_total[c] += class_mask.sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    writer.add_scalar('Loss/val', epoch_loss, epoch)
    writer.add_scalar('Accuracy/val', epoch_acc, epoch)
    
    # Log per-class accuracy
    for c in range(15):
        if class_total[c] > 0:
            class_acc = class_correct[c] / class_total[c]
            writer.add_scalar(f'Accuracy_class_{c}/val', class_acc, epoch)
            
    return epoch_loss, epoch_acc, all_preds, all_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='simclr_finetune')
    parser.add_argument('--encoder_path', type=str, required=True, help='Path to pretrained encoder weights')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder weights')
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
    
    # Data Loading
    labeled_df = pd.read_csv(r'data\labelled.csv')
    labeled_volume_dir = Path(r'data\volumes\volumes\labelled')
    
    train_df, val_df = train_test_split(labeled_df, test_size=0.5, stratify=labeled_df['label'], random_state=42)
    
    train_transform = get_train_transform()
    val_transform = get_val_transform()
    
    train_dataset = ForamDataset(train_df, labeled_volume_dir, transform=train_transform)
    val_dataset = ForamDataset(val_df, labeled_volume_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model
    model = resnet18_3d(num_classes=15)
    
    # Load pretrained encoder
    print(f"Loading pretrained encoder from {args.encoder_path}")
    # The saved state dict is from model.encoder, which is a ResNet3d
    # However, in SimCLR pretraining, we replaced fc with Identity
    # So the state dict won't have 'fc.weight' and 'fc.bias'
    # But our new model DOES have 'fc'
    
    state_dict = torch.load(args.encoder_path)
    
    # Filter out fc keys just in case (though they shouldn't be there)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc')}
    
    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {missing}") # Should be fc.weight, fc.bias
    print(f"Unexpected keys: {unexpected}") # Should be empty
    
    if args.freeze_encoder:
        print("Freezing encoder weights")
        for name, param in model.named_parameters():
            if not name.startswith('fc'):
                param.requires_grad = False
                
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, writer, epoch, scaler)
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device, writer, epoch)
        
        scheduler.step()
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), experiment_path / f'best_model_acc_val_epoch_{epoch}_{best_acc:.4f}.pth')
            print("Saved best model")
            
    print(f"Best Accuracy: {best_acc}")
    writer.close()

if __name__ == '__main__':
    main()
