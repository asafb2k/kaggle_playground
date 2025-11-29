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
from torch.utils.data import Dataset, DataLoader, Subset
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

def get_weak_transform(mean=0.15, std=0.25):
    return Compose([
        RandomFlip3D(p=0.5),
        Normalize(mean=mean, std=std)
    ])

def get_strong_transform(mean=0.15, std=0.25):
    return Compose([
        RandomFlip3D(p=0.5),
        RandomRotate90_3D(p=0.5),
        RandomIntensityScale(scale_limit=0.1, p=0.5),
        RandomNoise(std=0.02, p=0.5),
        RandomCutout3D(length=32, p=0.5),
        Normalize(mean=mean, std=std)
    ])

def get_val_transform(mean=0.15, std=0.25):
    return Compose([
        Normalize(mean=mean, std=std)
    ])

class ForamFixMatchDataset(Dataset):
    def __init__(self, data_frame, volume_dir, weak_transform=None, strong_transform=None):
        self.data_frame = data_frame
        self.volume_dir = volume_dir
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform
        
        self.file_paths = {}
        prefix = "foram_"
        for file in self.volume_dir.iterdir():
            if file.name.startswith(prefix):
                file_id = int(file.name.split('_')[1])
                self.file_paths[file_id] = self.volume_dir / file.name
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        file_id = int(self.data_frame.iloc[idx]['id'])
        file_path = self.file_paths.get(file_id)
        if not file_path:
            raise FileNotFoundError(f"Could not find file for ID {file_id}")
        
        volume = tifffile.imread(file_path)
        volume = volume.reshape(1, *volume.shape)
        volume = volume.astype(np.float32) / volume.max()
        
        # Apply transforms
        # We need to apply transforms to copies of the volume to avoid modifying the original if cached (though here we reload)
        # But importantly, weak and strong transforms are applied independently
        
        v_weak = volume.copy()
        if self.weak_transform:
            v_weak = self.weak_transform(v_weak)
            
        v_strong = volume.copy()
        if self.strong_transform:
            v_strong = self.strong_transform(v_strong)
            
        return torch.tensor(v_weak, dtype=torch.float32), torch.tensor(v_strong, dtype=torch.float32), file_id

class ForamLabeledDataset(Dataset):
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

def train_fixmatch(model, labeled_loader, unlabeled_loader, optimizer, device, writer, epoch, 
                   threshold=0.95, lambda_u=1.0, scaler=None):
    model.train()
    running_loss = 0.0
    running_loss_x = 0.0
    running_loss_u = 0.0
    mask_probs = 0.0
    
    # We need to iterate over both loaders. Since unlabeled is usually larger, we cycle labeled.
    labeled_iter = iter(labeled_loader)
    
    pbar = tqdm(unlabeled_loader, desc=f"Train Epoch {epoch}")
    for batch_idx, (inputs_u_w, inputs_u_s, _) in enumerate(pbar):
        try:
            inputs_x, targets_x = next(labeled_iter)
        except StopIteration:
            labeled_iter = iter(labeled_loader)
            inputs_x, targets_x = next(labeled_iter)
            
        inputs_x = inputs_x.to(device)
        targets_x = targets_x.to(device)
        inputs_u_w = inputs_u_w.to(device)
        inputs_u_s = inputs_u_s.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision
        if scaler:
            with autocast():
                # Labeled data forward
                logits_x = model(inputs_x)
                Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
                
                # Unlabeled data forward
                # Generate pseudo-labels from weak augmentation
                with torch.no_grad():
                    logits_u_w = model(inputs_u_w)
                    probs_u_w = torch.softmax(logits_u_w, dim=1)
                    max_probs, p_targets_u = torch.max(probs_u_w, dim=1)
                    mask = max_probs.ge(threshold).float()
                
                # Strong augmentation forward
                logits_u_s = model(inputs_u_s)
                Lu = (F.cross_entropy(logits_u_s, p_targets_u, reduction='none') * mask).mean()
                
                loss = Lx + lambda_u * Lu
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular precision
            logits_x = model(inputs_x)
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')
            
            with torch.no_grad():
                logits_u_w = model(inputs_u_w)
                probs_u_w = torch.softmax(logits_u_w, dim=1)
                max_probs, p_targets_u = torch.max(probs_u_w, dim=1)
                mask = max_probs.ge(threshold).float()
            
            logits_u_s = model(inputs_u_s)
            Lu = (F.cross_entropy(logits_u_s, p_targets_u, reduction='none') * mask).mean()
            
            loss = Lx + lambda_u * Lu
            loss.backward()
            optimizer.step()
            
        running_loss += loss.item()
        running_loss_x += Lx.item()
        running_loss_u += Lu.item()
        mask_probs += mask.mean().item()
        
        pbar.set_postfix({'Lx': Lx.item(), 'Lu': Lu.item(), 'Mask': mask.mean().item()})
        
    # Log metrics
    steps = len(unlabeled_loader)
    # Use 'Loss/train' for the main loss to match previous experiments
    writer.add_scalar('Loss/train', running_loss / steps, epoch)
    # Log components separately
    writer.add_scalar('FixMatch/Loss_x', running_loss_x / steps, epoch)
    writer.add_scalar('FixMatch/Loss_u', running_loss_u / steps, epoch)
    writer.add_scalar('FixMatch/Mask_pct', mask_probs / steps, epoch)
    
    return running_loss / steps

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
    
    # Log metrics to TensorBoard (matching previous format)
    writer.add_scalar('Loss/val', epoch_loss, epoch)
    writer.add_scalar('Accuracy/val', epoch_acc, epoch)
    
    # Log per-class accuracy
    for c in range(15):
        if class_total[c] > 0:
            class_acc = class_correct[c] / class_total[c]
            writer.add_scalar(f'Accuracy_class_{c}/val', class_acc, epoch)
            
    # Create and log confusion matrix as figure
    cm = confusion_matrix(all_labels, all_preds)
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    writer.add_figure('Confusion_Matrix', fig, epoch)
    plt.close(fig)
    
    return epoch_loss, epoch_acc, all_preds, all_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='fixmatch_resnet18')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001) # Higher LR for FixMatch
    parser.add_argument('--threshold', type=float, default=0.95)
    parser.add_argument('--lambda_u', type=float, default=1.0)
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
    unlabeled_df = pd.read_csv(r'data\unlabelled.csv')
    
    train_df, val_df = train_test_split(labeled_df, test_size=0.5, stratify=labeled_df['label'], random_state=42)
    
    labeled_volume_dir = Path(r'data\volumes\volumes\labelled')
    unlabeled_volume_dir = Path(r'data\volumes\volumes\unlabelled')
    
    # Transforms
    weak_transform = get_weak_transform()
    strong_transform = get_strong_transform()
    val_transform = get_val_transform()
    
    # Datasets
    train_dataset = ForamLabeledDataset(train_df, labeled_volume_dir, transform=weak_transform)
    val_dataset = ForamLabeledDataset(val_df, labeled_volume_dir, transform=val_transform)
    
    # Use subset of unlabeled for speed if needed, but FixMatch benefits from more data
    # Let's use 2000 samples
    subset_indices = np.random.choice(len(unlabeled_df), 2000, replace=False)
    unlabeled_subset_df = unlabeled_df.iloc[subset_indices]
    
    unlabeled_dataset = ForamFixMatchDataset(unlabeled_subset_df, unlabeled_volume_dir, 
                                             weak_transform=weak_transform, 
                                             strong_transform=strong_transform)
    
    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    
    # Model
    model = resnet18_3d(num_classes=15).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss = train_fixmatch(model, train_loader, unlabeled_loader, optimizer, device, writer, epoch, 
                                    threshold=args.threshold, lambda_u=args.lambda_u, scaler=scaler)
        
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device, writer, epoch)
        
        # Log learning rate
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), experiment_path / f'best_model_acc_val_epoch_{epoch}_{best_acc:.4f}.pth')
            print("Saved best model")
            
    print(f"Best Accuracy: {best_acc}")
    writer.close()

if __name__ == '__main__':
    main()
