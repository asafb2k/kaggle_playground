import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime
import random

# Import custom modules
from models import resnet18_3d, resnet34_3d
from transforms import Compose, RandomFlip3D, RandomRotate90_3D, RandomIntensityScale, RandomNoise, RandomCutout3D, Normalize

# ... (rest of imports)

# ... (EnsembleTeacher class)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='noisy_student')
    parser.add_argument('--teacher_paths', nargs='+', required=True, help='Paths to teacher model weights')
    parser.add_argument('--student_arch', type=str, default='resnet18', choices=['resnet18', 'resnet34'], help='Student model architecture')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--threshold', type=float, default=0.7)
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
    
    labeled_volume_dir = Path(r'data\volumes\volumes\labelled')
    unlabeled_volume_dir = Path(r'data\volumes\volumes\unlabelled')
    
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(labeled_df, test_size=0.5, stratify=labeled_df['label'], random_state=42)
    
    # 1. Load Teacher (Ensemble)
    print(f"Loading {len(args.teacher_paths)} teacher models...")
    teacher = EnsembleTeacher(args.teacher_paths, device)
    
    # 2. Generate Pseudo-labels
    val_transform = get_val_transform()
    unlabeled_dataset = UnlabeledDataset(unlabeled_df, unlabeled_volume_dir, transform=val_transform)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Modified generate_pseudo_labels to handle EnsembleTeacher output (probabilities)
    # The original function expected logits.
    # Let's redefine generate_pseudo_labels locally or update it.
    
    teacher.eval()
    pseudo_labels = {}
    print("Generating pseudo-labels...")
    
    with torch.no_grad():
        for inputs, ids in tqdm(unlabeled_loader, desc="Pseudo-labeling"):
            inputs = inputs.to(device)
            
            # TTA with Ensemble
            # EnsembleTeacher already returns probabilities
            probs = teacher(inputs)
            
            # Additional TTA (flips)
            probs += teacher(torch.flip(inputs, [2]))
            probs += teacher(torch.flip(inputs, [3]))
            probs += teacher(torch.flip(inputs, [4]))
            probs /= 4.0
            
            max_probs, preds = torch.max(probs, dim=1)
            
            # Store confident predictions
            for i in range(len(ids)):
                if max_probs[i] > args.threshold:
                    pseudo_labels[int(ids[i])] = int(preds[i])
                    
    print(f"Generated {len(pseudo_labels)} pseudo-labels")
    
    # 3. Train Student
    print(f"Training student model ({args.student_arch})...")
    student_transform = get_student_transform()
    
    # Create combined dataset
    train_dataset = NoisyStudentDataset(
        train_df, unlabeled_df, 
        labeled_volume_dir, unlabeled_volume_dir, 
        pseudo_labels, 
        transform=student_transform
    )
    
    val_dataset = NoisyStudentDataset(
        val_df, pd.DataFrame(columns=['id']), 
        labeled_volume_dir, unlabeled_volume_dir, 
        {}, 
        transform=val_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    if args.student_arch == 'resnet18':
        student = resnet18_3d(num_classes=15).to(device)
    elif args.student_arch == 'resnet34':
        student = resnet34_3d(num_classes=15).to(device)

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

def get_student_transform(mean=0.15, std=0.25):
    # Strong augmentation for student
    return Compose([
        RandomFlip3D(p=0.5),
        RandomRotate90_3D(p=0.5),
        RandomIntensityScale(scale_limit=0.15, p=0.5),
        RandomNoise(std=0.03, p=0.5),
        RandomCutout3D(length=32, p=0.5),
        Normalize(mean=mean, std=std)
    ])

def get_val_transform(mean=0.15, std=0.25):
    return Compose([
        Normalize(mean=mean, std=std)
    ])

class NoisyStudentDataset(Dataset):
    def __init__(self, labeled_df, unlabeled_df, labeled_dir, unlabeled_dir, pseudo_labels, transform=None):
        self.labeled_df = labeled_df
        self.unlabeled_df = unlabeled_df
        self.labeled_dir = labeled_dir
        self.unlabeled_dir = unlabeled_dir
        self.pseudo_labels = pseudo_labels # Dict mapping file_id -> label
        self.transform = transform
        
        self.file_paths = {}
        
        # Index labeled files
        prefix_l = "labelled_foram_"
        for file in self.labeled_dir.iterdir():
            if file.name.startswith(prefix_l):
                file_id = int(file.name.split('_')[2])
                self.file_paths[f"l_{file_id}"] = self.labeled_dir / file.name
                
        # Index unlabeled files
        prefix_u = "foram_"
        for file in self.unlabeled_dir.iterdir():
            if file.name.startswith(prefix_u):
                file_id = int(file.name.split('_')[1])
                self.file_paths[f"u_{file_id}"] = self.unlabeled_dir / file.name
                
        # Create combined list of samples
        self.samples = []
        
        # Add labeled samples
        for idx in range(len(labeled_df)):
            file_id = int(labeled_df.iloc[idx]['id'].split('_')[1])
            label = labeled_df.iloc[idx]['label']
            self.samples.append({
                'id': f"l_{file_id}",
                'label': label,
                'is_labeled': True
            })
            
        # Add pseudo-labeled samples
        for idx in range(len(unlabeled_df)):
            file_id = int(unlabeled_df.iloc[idx]['id'])
            if file_id in self.pseudo_labels:
                self.samples.append({
                    'id': f"u_{file_id}",
                    'label': self.pseudo_labels[file_id],
                    'is_labeled': False
                })
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        file_path = self.file_paths[sample['id']]
        label = sample['label']
        
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
    
    for inputs, labels in tqdm(dataloader, desc="Training Student"):
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
            
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    writer.add_scalar('Loss/val', epoch_loss, epoch)
    writer.add_scalar('Accuracy/val', epoch_acc, epoch)
    
    return epoch_loss, epoch_acc

class EnsembleTeacher(nn.Module):
    def __init__(self, model_paths, device):
        super().__init__()
        self.models = nn.ModuleList()
        for path in model_paths:
            model = resnet18_3d(num_classes=15)
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            model.eval()
            self.models.append(model)
            
    def forward(self, x):
        # Average probabilities
        probs_sum = None
        for model in self.models:
            outputs = model(x)
            probs = F.softmax(outputs, dim=1)
            if probs_sum is None:
                probs_sum = probs
            else:
                probs_sum += probs
        return probs_sum / len(self.models)

class UnlabeledDataset(Dataset):
    def __init__(self, data_frame, volume_dir, transform=None):
        self.data_frame = data_frame
        self.volume_dir = volume_dir
        self.transform = transform
        
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
        
        volume = tifffile.imread(file_path)
        volume = volume.reshape(1, *volume.shape)
        volume = volume.astype(np.float32) / volume.max()
        
        if self.transform:
            volume = self.transform(volume)
            
        return torch.tensor(volume, dtype=torch.float32), file_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='noisy_student')
    parser.add_argument('--teacher_paths', nargs='+', required=True, help='Paths to teacher model weights')
    parser.add_argument('--student_arch', type=str, default='resnet18', choices=['resnet18', 'resnet34'], help='Student model architecture')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--threshold', type=float, default=0.7)
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
    
    labeled_volume_dir = Path(r'data\volumes\volumes\labelled')
    unlabeled_volume_dir = Path(r'data\volumes\volumes\unlabelled')
    
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(labeled_df, test_size=0.5, stratify=labeled_df['label'], random_state=42)
    
    # 1. Load Teacher (Ensemble)
    print(f"Loading {len(args.teacher_paths)} teacher models...")
    teacher = EnsembleTeacher(args.teacher_paths, device)
    
    # 2. Generate Pseudo-labels
    val_transform = get_val_transform()
    unlabeled_dataset = UnlabeledDataset(unlabeled_df, unlabeled_volume_dir, transform=val_transform)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    teacher.eval()
    pseudo_labels = {}
    print("Generating pseudo-labels...")
    
    with torch.no_grad():
        for inputs, ids in tqdm(unlabeled_loader, desc="Pseudo-labeling"):
            inputs = inputs.to(device)
            
            # TTA with Ensemble
            # EnsembleTeacher already returns probabilities
            probs = teacher(inputs)
            
            # Additional TTA (flips)
            probs += teacher(torch.flip(inputs, [2]))
            probs += teacher(torch.flip(inputs, [3]))
            probs += teacher(torch.flip(inputs, [4]))
            probs /= 4.0
            
            max_probs, preds = torch.max(probs, dim=1)
            
            # Store confident predictions
            for i in range(len(ids)):
                if max_probs[i] > args.threshold:
                    pseudo_labels[int(ids[i])] = int(preds[i])
                    
    print(f"Generated {len(pseudo_labels)} pseudo-labels")
    
    # 3. Train Student
    print(f"Training student model ({args.student_arch})...")
    student_transform = get_student_transform()
    
    # Create combined dataset
    train_dataset = NoisyStudentDataset(
        train_df, unlabeled_df, 
        labeled_volume_dir, unlabeled_volume_dir, 
        pseudo_labels, 
        transform=student_transform
    )
    
    val_dataset = NoisyStudentDataset(
        val_df, pd.DataFrame(columns=['id']), 
        labeled_volume_dir, unlabeled_volume_dir, 
        {}, 
        transform=val_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    if args.student_arch == 'resnet18':
        student = resnet18_3d(num_classes=15).to(device)
    elif args.student_arch == 'resnet34':
        student = resnet34_3d(num_classes=15).to(device)
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(student, train_loader, criterion, optimizer, device, writer, epoch, scaler)
        val_loss, val_acc = validate(student, val_loader, criterion, device, writer, epoch)
        
        scheduler.step()
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(student.state_dict(), experiment_path / f'best_student_acc_val_epoch_{epoch}_{best_acc:.4f}.pth')
            print("Saved best student model")
            
    print(f"Best Student Accuracy: {best_acc}")
    writer.close()

if __name__ == '__main__':
    main()
