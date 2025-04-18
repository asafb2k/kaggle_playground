import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm
import time

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Define the CNN model for 3D volumes
class Foram3DCNN(nn.Module):
    def __init__(self, num_classes=15):  # Changed to 15 classes (14 + unknown)
        super(Foram3DCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # Output: [16, 64, 64, 64]
        
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # Output: [32, 32, 32, 32]
        
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)  # Output: [64, 16, 16, 16]
        
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(128)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)  # Output: [128, 8, 8, 8]
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 8 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Convolutional layers with batch normalization and pooling
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten for fully connected layers
        x = x.view(-1, 128 * 8 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Dataset class for loading labeled 3D volumes
class ForamDataset(Dataset):
    def __init__(self, data_frame, volume_dir):
        self.data_frame = data_frame
        self.volume_dir = volume_dir
        
        # Pre-find all file paths to speed up loading
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
        
        # Get pre-found file path
        file_path = self.file_paths.get(file_id)
        if not file_path:
            raise FileNotFoundError(f"Could not find file for ID {file_id}")
        
        # Load volume
        volume = tifffile.imread(file_path)
        
        # Add channel dimension and normalize
        volume = volume.reshape(1, *volume.shape)
        volume = volume.astype(np.float32) / volume.max()  # Normalize to [0,1]
        
        volume_tensor = torch.tensor(volume, dtype=torch.float32)
        
        return volume_tensor, label

# Dataset class for loading unlabeled 3D volumes
class ForamUnlabeledDataset(Dataset):
    def __init__(self, data_frame, volume_dir):
        self.data_frame = data_frame
        self.volume_dir = volume_dir
        
        # Pre-find all file paths to speed up loading
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
        
        # Get pre-found file path
        file_path = self.file_paths.get(file_id)
        if not file_path:
            raise FileNotFoundError(f"Could not find file for ID {file_id}")
        
        # Load volume
        volume = tifffile.imread(file_path)
        
        # Add channel dimension and normalize
        volume = volume.reshape(1, *volume.shape)
        volume = volume.astype(np.float32) / volume.max()  # Normalize to [0,1]
        
        volume_tensor = torch.tensor(volume, dtype=torch.float32)
        
        return volume_tensor, file_id

# Training and validation functions
def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        if scaler:  # Use mixed precision training if scaler is provided
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            # Scale loss and do backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular training
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # For confusion matrix
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels

# Semi-supervised learning with pseudo-labeling
def pseudo_label_epoch(model, labeled_loader, unlabeled_loader, criterion, optimizer, device, 
                       pseudo_threshold=0.8, pseudo_weight=0.5, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # First, train on labeled data
    for inputs, labels in tqdm(labeled_loader, desc="Training on labeled data"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        if scaler:  # Use mixed precision training if scaler is provided
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            # Scale loss and do backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular training
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Then, use pseudo-labeling on unlabeled data
    model.eval()  # Switch to eval mode to generate pseudo-labels
    pseudo_labeled = 0
    
    with torch.no_grad():
        for inputs, _ in tqdm(unlabeled_loader, desc="Generating pseudo-labels"):
            inputs = inputs.to(device)
            
            # Generate pseudo-labels
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            max_probs, pseudo_labels = torch.max(probs, dim=1)
            
            # Only use confident predictions
            mask = max_probs > pseudo_threshold
            
            # Skip if no confident predictions
            if not mask.any():
                continue
                
            # Use confident pseudo-labels for training
            confident_inputs = inputs[mask]
            confident_labels = pseudo_labels[mask]
            
            # Count how many pseudo-labels we're using
            pseudo_labeled += mask.sum().item()
    
    # Now train on pseudo-labeled data
    if pseudo_labeled > 0:
        model.train()  # Switch back to train mode
        
        for inputs, _ in tqdm(unlabeled_loader, desc="Training on pseudo-labeled data"):
            inputs = inputs.to(device)
            
            # Generate pseudo-labels
            with torch.no_grad():
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
                max_probs, pseudo_labels = torch.max(probs, dim=1)
                
                # Only use confident predictions
                mask = max_probs > pseudo_threshold
                
                # Skip if no confident predictions
                if not mask.any():
                    continue
            
            # Use confident pseudo-labels for training
            confident_inputs = inputs[mask]
            confident_labels = pseudo_labels[mask]
            
            # Train on pseudo-labeled data
            optimizer.zero_grad()
            
            if scaler:
                with autocast():
                    outputs_pl = model(confident_inputs)
                    # Apply a weight to pseudo-labeled loss (typically lower than labeled data)
                    loss_pl = criterion(outputs_pl, confident_labels) * pseudo_weight
                    
                # Scale loss and do backward pass
                scaler.scale(loss_pl).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs_pl = model(confident_inputs)
                loss_pl = criterion(outputs_pl, confident_labels) * pseudo_weight
                loss_pl.backward()
                optimizer.step()
            
            # Update metrics (but don't count pseudo-labeled samples in accuracy)
            running_loss += loss_pl.item() * confident_inputs.size(0)
    
    print(f"Used {pseudo_labeled} pseudo-labeled samples")
    
    # Return the metrics from labeled data only for consistent monitoring
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def main():
    experiment_name = 'semi_supervised_learning'
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load labeled data
    labeled_data_path = Path(r'data\labelled.csv')  # Update with your path
    labeled_volume_dir = Path(r'data\volumes\volumes\labelled')  # Update with your path
    
    labeled_df = pd.read_csv(labeled_data_path)
    print(f"Loaded {len(labeled_df)} labeled samples")
    
    # Load unlabeled data
    unlabeled_data_path = Path(r'data\unlabelled.csv')  # Update with your path
    unlabeled_volume_dir = Path(r'data\volumes\volumes\unlabelled')  # Update with your path
    
    unlabeled_df = pd.read_csv(unlabeled_data_path)
    print(f"Loaded {len(unlabeled_df)} unlabeled samples")
    
    # Split labeled data into train and validation sets
    train_df, val_df = train_test_split(labeled_df, test_size=0.35, stratify=labeled_df['label'], random_state=42)
    
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
    
    # Create datasets
    train_dataset = ForamDataset(train_df, labeled_volume_dir)
    val_dataset = ForamDataset(val_df, labeled_volume_dir)
    unlabeled_dataset = ForamUnlabeledDataset(unlabeled_df, unlabeled_volume_dir)
    
    # For faster experimentation, use a subset of unlabeled data
    # Comment out this section if you want to use all unlabeled data
    subset_size = 1000  # Adjust as needed based on your GPU memory
    subset_indices = np.random.choice(len(unlabeled_dataset), subset_size, replace=False)
    unlabeled_subset = Subset(unlabeled_dataset, subset_indices)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
    # Use the subset for faster training during development
    unlabeled_loader = DataLoader(unlabeled_subset, batch_size=2, shuffle=True, num_workers=0)
    # For final training, use all unlabeled data:
    # unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=2, shuffle=True, num_workers=0)
    
    # Initialize model, loss function, and optimizer
    model = Foram3DCNN(num_classes=15)  # 15 classes including unknown class
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Use mixed precision training if available
    use_amp = torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    
    # Training loop
    num_epochs = 100
    best_val_acc = 0.0
    
    # Hyperparameters for pseudo-labeling
    pseudo_start_epoch = 5  # Start pseudo-labeling after this epoch
    pseudo_threshold = 0.85  # Confidence threshold for pseudo-labels
    pseudo_weight = 0.5  # Weight for pseudo-labeled loss
    
    # For tracking metrics
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train with pseudo-labeling
        if epoch >= pseudo_start_epoch:
            train_loss, train_acc = pseudo_label_epoch(
                model, train_loader, unlabeled_loader, criterion, optimizer, device, 
                pseudo_threshold=pseudo_threshold, pseudo_weight=pseudo_weight, scaler=scaler
            )
        else:
            # Regular supervised training for initial epochs
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, scaler
            )
            
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, all_preds, all_labels = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        experiments_directory = Path("experiments")
        experiment_path = experiments_directory / experiment_name
        os.makedirs(experiment_path, exist_ok=True)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_save_path = experiment_path / f'best_model_acc_val_{best_val_acc}.pth'
            torch.save(model.state_dict(), model_save_path.as_posix())
            print("Saved best model checkpoint")
            
            # Generate detailed metrics for best model
            print("\nClassification Report:")
            print(classification_report(all_labels, all_preds))
            
            # Calculate confusion matrix
            cm = confusion_matrix(all_labels, all_preds)
            
            # Plot and save confusion matrix
            plt.figure(figsize=(10, 8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            confusion_matrix_path = experiment_path / 'confusion_matrix.png'
            plt.savefig(confusion_matrix_path.as_posix())
            plt.close()
    
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time / 60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Plot training and validation metrics
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    fig_save_path = experiment_path / 'training_metrics.png'
    plt.savefig(fig_save_path.as_posix())
    plt.close()
    
    print("Training and validation plots saved.")

    # Generate final submission
    generate_submission(model, unlabeled_dataset, device, experiment_path)

def generate_submission(model, unlabeled_dataset, device, experiment_path):
    """Generate submission file for the competition"""
    # Use all unlabeled data for final prediction
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    model.eval()
    all_ids = []
    all_preds = []
    
    confidence_threshold = 0.7  # Threshold for unknown class
    
    with torch.no_grad():
        for inputs, ids in tqdm(unlabeled_loader, desc="Generating predictions"):
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get probabilities
            probs = F.softmax(outputs, dim=1)
            
            # For 15-class model, handle unknown class differently
            # First 14 classes (0-13) are the known foram types
            # Class 14 is the unknown class
            
            # Get max probability and prediction for known classes only
            known_probs = probs[:, :14]  # Only consider classes 0-13
            max_probs, preds = torch.max(known_probs, dim=1)
            
            # If confidence is below threshold, assign to unknown class (14)
            unknown_mask = max_probs < confidence_threshold
            preds[unknown_mask] = 14
            
            # Store results
            all_ids.extend(ids.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': all_ids,
        'label': all_preds
    })
    
    # Sort by ID to match the expected order
    submission_df = submission_df.sort_values('id')
    
    # Save submission file
    submission_path = experiment_path / 'submission.csv'
    submission_df.to_csv(submission_path, index=False)
    
    print(f"Submission file saved to {submission_path}")
    
    # Print prediction statistics
    print("\nClass distribution in predictions:")
    value_counts = submission_df['label'].value_counts().sort_index()
    for class_id, count in value_counts.items():
        print(f"Class {class_id}: {count} samples ({count/len(submission_df)*100:.2f}%)")

if __name__ == "__main__":
    main()