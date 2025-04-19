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

# Import TensorBoard
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import time
import datetime

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
def train_epoch(model, dataloader, criterion, optimizer, device, writer, epoch, scaler=None):
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
    
    # Log metrics to TensorBoard
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/train', epoch_acc, epoch)
    
    # Log learning rate
    for param_group in optimizer.param_groups:
        writer.add_scalar('Learning_rate', param_group['lr'], epoch)
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device, writer, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # For confusion matrix
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
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Track per-class accuracy
            for c in range(15):
                class_mask = (labels == c)
                class_correct[c] += (predicted[class_mask] == c).sum().item()
                class_total[c] += class_mask.sum().item()
            
            # Store predictions and labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    # Log metrics to TensorBoard
    writer.add_scalar('Loss/val', epoch_loss, epoch)
    writer.add_scalar('Accuracy/val', epoch_acc, epoch)
    
    # Log per-class accuracy to TensorBoard
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

def pseudo_label_epoch(model, labeled_loader, unlabeled_loader, criterion, optimizer, device, 
                      writer, epoch, pseudo_threshold=0.99, pseudo_weight=0.75, scaler=None):
    # First, collect pseudo-labels in evaluation mode
    model.eval()
    
    # Collect confident samples directly
    confident_inputs_list = []
    confident_labels_list = []
    pseudo_class_counts = np.zeros(15)
    max_confidence_in_this_epoch = 0
    
    with torch.no_grad():
        for inputs, ids in tqdm(unlabeled_loader, desc="Generating pseudo-labels"):
            inputs = inputs.to(device)
            
            # Generate pseudo-labels
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            max_probs, pseudo_labels = torch.max(probs, dim=1)
            max_confidence_in_this_epoch = max(max_probs.max().item(), max_confidence_in_this_epoch)
            
            # Only use confident predictions
            mask = max_probs > pseudo_threshold
            
            # Skip if no confident predictions
            if not mask.any():
                continue
                
            # Track class distribution of pseudo-labels
            for c in range(15):
                pseudo_class_counts[c] += (pseudo_labels[mask] == c).sum().item()
            
            # Collect confident samples
            confident_inputs_list.append(inputs[mask].cpu())
            confident_labels_list.append(pseudo_labels[mask].cpu())
    
    # Combine all confident samples
    pseudo_labeled = 0
    if confident_inputs_list:
        confident_inputs = torch.cat(confident_inputs_list, dim=0)
        confident_labels = torch.cat(confident_labels_list, dim=0)
        pseudo_labeled = confident_labels.shape[0]
    
    print(f"Max confidence in this epoch: {max_confidence_in_this_epoch:.4f}")
    print(f"Generated {pseudo_labeled} confident pseudo-labels")
    writer.add_scalar('Pseudo/num_pseudo_labels', pseudo_labeled, epoch)
    
    # SWITCH TO TRAINING MODE FOR THE COMBINED TRAINING
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # If we have pseudo-labeled data, combine it with labeled data in each batch
    if pseudo_labeled > 0:
        # Create a dataset from the confident samples
        confident_dataset = torch.utils.data.TensorDataset(confident_inputs, confident_labels)
        confident_loader = DataLoader(
            confident_dataset, 
            batch_size=labeled_loader.batch_size, 
            shuffle=True,
            num_workers=0
        )
        
        # First, train on labeled data
        for inputs, labels in tqdm(labeled_loader, desc="Training on labeled data"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                # Scale loss and do backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Then, train on pseudo-labeled data
        pseudo_loss_sum = 0.0
        for inputs, labels in tqdm(confident_loader, desc="Training on pseudo-labeled data"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast(device_type='cuda'):
                    outputs_pl = model(inputs)
                    loss_pl = criterion(outputs_pl, labels) * pseudo_weight
                    
                scaler.scale(loss_pl).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs_pl = model(inputs)
                loss_pl = criterion(outputs_pl, labels) * pseudo_weight
                loss_pl.backward()
                optimizer.step()
            
            # Update metrics for pseudo-labeled data
            pseudo_loss_sum += loss_pl.item() * inputs.size(0)
            
            # Include pseudo-labeled samples in overall metrics
            running_loss += loss_pl.item() * inputs.size(0)
            _, predicted = torch.max(outputs_pl.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Log pseudo-labeling metrics to TensorBoard
        writer.add_scalar('Pseudo/samples_used', pseudo_labeled, epoch)
        writer.add_scalar('Pseudo/avg_loss', pseudo_loss_sum / pseudo_labeled, epoch)
        
        # Log class distribution of pseudo-labels
        for c in range(15):
            writer.add_scalar(f'Pseudo/class_{c}_count', pseudo_class_counts[c], epoch)
    else:
        # If no pseudo-labels, just train on labeled data
        for inputs, labels in tqdm(labeled_loader, desc="Training on labeled data only"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                # Scale loss and do backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate epoch metrics
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    # Log metrics to TensorBoard
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/train', epoch_acc, epoch)
    
    return epoch_loss, epoch_acc
def main():
    # Create a unique run name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f'semi_supervised_learning_{timestamp}'
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories for experiment
    experiments_directory = Path("experiments")
    experiment_path = experiments_directory / experiment_name
    tensorboard_path = experiment_path / 'tensorboard'
    os.makedirs(experiment_path, exist_ok=True)
    os.makedirs(tensorboard_path, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=tensorboard_path)
    
    # Save experiment configuration for tracking
    config = {
        'experiment_name': experiment_name,
        'device': str(device),
        'batch_size': 16,
        'learning_rate': 0.000001,
        'weight_decay': 1e-5,
        'num_epochs': 1000,
        'pseudo_start_epoch': 30,
        'confidence_threshold': 0.25,
        'pseudo_initial_threshold': 0.995,  # Start with higher confidence
        'pseudo_final_threshold': 0.8,    # End with lower threshold
        'pseudo_initial_weight': 0.3,      # Start with lower weight
        'pseudo_final_weight': 0.7,        # End with higher weight
    }
    
    # Save config to experiment directory
    with open(experiment_path / 'config.txt', 'w') as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
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
    train_df, val_df = train_test_split(labeled_df, test_size=0.5, stratify=labeled_df['label'], random_state=42)
    
    print(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")
    
    # Log dataset information to TensorBoard
    writer.add_text('Data/total_labeled', str(len(labeled_df)))
    writer.add_text('Data/train_labeled', str(len(train_df)))
    writer.add_text('Data/val_labeled', str(len(val_df)))
    writer.add_text('Data/unlabeled', str(len(unlabeled_df)))
    
    # Create datasets
    train_dataset = ForamDataset(train_df, labeled_volume_dir)
    val_dataset = ForamDataset(val_df, labeled_volume_dir)
    unlabeled_dataset = ForamUnlabeledDataset(unlabeled_df, unlabeled_volume_dir)
    
    # For faster experimentation, use a subset of unlabeled data
    # Comment out this section if you want to use all unlabeled data
    subset_size = 1600  # Adjust as needed based on your GPU memory
    subset_indices = np.random.choice(len(unlabeled_dataset), subset_size, replace=False)
    unlabeled_subset = Subset(unlabeled_dataset, subset_indices)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    # Use the subset for faster training during development
    unlabeled_loader = DataLoader(unlabeled_subset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    # For final training, use all unlabeled data:
    # unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    
    # Initialize model, loss function, and optimizer
    model = Foram3DCNN(num_classes=15)  # 15 classes including unknown class
    model = model.to(device)
    
    # Log model graph to TensorBoard (requires a sample input)
    sample_input = torch.zeros(1, 1, 128, 128, 128, device=device)
    writer.add_graph(model, sample_input)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    warmup_epochs = int(0.01 * config['num_epochs'])  # 10% of epochs for warmup
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.2,  # Start at 10% of the base learning rate
        end_factor=1.0,    # End at the full base learning rate
        total_iters=warmup_epochs
    )

    # Then, define the main scheduler that will run after warmup
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs'] - warmup_epochs,
        eta_min=1e-7  # Minimum learning rate at the end
    )

    # And use a sequential scheduler to chain them
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_epochs]
    )
    
    # Use mixed precision training if available
    use_amp = torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    
    # Training loop
    num_epochs = config['num_epochs']
    best_val_acc = 0.0
    
    # Hyperparameters for pseudo-labeling
    pseudo_start_epoch = config['pseudo_start_epoch']
    
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
            # Calculate dynamic threshold and weight based on progress
            progress = min(1.0, (epoch - pseudo_start_epoch) / (num_epochs - pseudo_start_epoch))
            current_threshold = config['pseudo_initial_threshold'] - progress * (config['pseudo_initial_threshold'] - config['pseudo_final_threshold'])
            current_weight = config['pseudo_initial_weight'] + progress * (config['pseudo_final_weight'] - config['pseudo_initial_weight'])
            writer.add_scalar('Pseudo/threshold', current_threshold, epoch)
            
            train_loss, train_acc = pseudo_label_epoch(
                model, train_loader, unlabeled_loader, criterion, optimizer, device, 
                writer, epoch, pseudo_threshold=current_threshold, pseudo_weight=current_weight, scaler=scaler
            )
        else:
            # Regular supervised training for initial epochs
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, writer, epoch, scaler
            )
            
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, all_preds, all_labels = validate(model, val_loader, criterion, device, writer, epoch)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_save_path = experiment_path / f'best_model_acc_val_epoch_{epoch}_{best_val_acc:.4f}.pth'
            torch.save(model.state_dict(), model_save_path.as_posix())
            print("Saved best model checkpoint")
            
            # Update best model info in TensorBoard
            writer.add_text('Model/best_epoch', str(epoch), epoch)
            writer.add_text('Model/best_accuracy', f"{best_val_acc:.4f}", epoch)
            
            # Generate detailed metrics for best model
            print("\nClassification Report:")
            report = classification_report(all_labels, all_preds)
            print(report)
            writer.add_text('Metrics/classification_report', report.replace('\n', '  \n'), epoch)
            
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
    
    # Log final training time
    writer.add_text('Training/time_minutes', f"{elapsed_time / 60:.2f}")
    writer.add_text('Training/best_accuracy', f"{best_val_acc:.4f}")
    
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
    generate_submission(model, unlabeled_dataset, device, experiment_path, config['confidence_threshold'])
    
    # Close TensorBoard writer
    writer.close()
    
    print(f"TensorBoard logs saved to {tensorboard_path}")
    print(f"To view TensorBoard, run: tensorboard --logdir={tensorboard_path}")

def generate_submission(model, unlabeled_dataset, device, experiment_path, confidence_threshold=0.7):
    """Generate submission file for the competition"""
    # Use all unlabeled data for final prediction
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    model.eval()
    all_ids = []
    all_preds = []
    
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
        percentage = count/len(submission_df)*100
        print(f"Class {class_id}: {count} samples ({percentage:.2f}%)")
        
        # Save this information to a text file
        with open(experiment_path / 'submission_stats.txt', 'a') as f:
            f.write(f"Class {class_id}: {count} samples ({percentage:.2f}%)\n")

def generate_submission_without_training(confidence_threshold):
    """
    Load the best model from a previous experiment and generate a submission file
    without training the model again.
    """
    import glob
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Find the latest experiment folder
    experiments_directory = Path("experiments")
    experiment_folders = list(experiments_directory.glob("semi_supervised_learning_*"))
    
    if not experiment_folders:
        raise ValueError("No experiment folders found. Please run training first.")
    
    # Sort by creation time (newest first)
    latest_experiment = max(experiment_folders, key=lambda x: os.path.getmtime(x))
    print(f"Using latest experiment: {latest_experiment}")
    
    # Load the configuration
    config = {}
    config_path = latest_experiment / 'config.txt'
    if config_path.exists():
        with open(config_path, 'r') as f:
            for line in f:
                key, value = line.strip().split(': ')
                config[key] = eval(value) if value.lower() in ['true', 'false'] or value.replace('.', '').replace('-', '').replace('e', '').isdigit() else value
        print(f"Loaded configuration from {config_path}")

    # Find the best model checkpoint
    model_files = list(latest_experiment.glob('best_model_acc_val_epoch_*.pth'))
    if not model_files:
        raise ValueError(f"No model checkpoints found in {latest_experiment}")
    
    # Sort by validation accuracy (highest first)
    best_model_path = max(model_files, key=lambda x: float(str(x).split('_')[-1].replace('.pth', '')))
    print(f"Using best model: {best_model_path}")
    
    # Initialize the model
    model = Foram3DCNN(num_classes=15)
    model = model.to(device)
    
    # Load model weights
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Loaded model weights from {best_model_path}")
    
    # Load unlabeled data
    unlabeled_data_path = Path(r'data\unlabelled.csv')  # Update with your path
    unlabeled_volume_dir = Path(r'data\volumes\volumes\unlabelled')  # Update with your path
    
    unlabeled_df = pd.read_csv(unlabeled_data_path)
    print(f"Loaded {len(unlabeled_df)} unlabeled samples")
    
    # Create dataset and dataloader
    unlabeled_dataset = ForamUnlabeledDataset(unlabeled_df, unlabeled_volume_dir)
    
    # Generate submission
    print(f'runing with confidence threshold: {confidence_threshold}')
    generate_submission(model, unlabeled_dataset, device, latest_experiment, confidence_threshold)
    
    print(f"Submission file generated in {latest_experiment}")


if __name__ == "__main__":
    main()
    # generate_submission_without_training(confidence_threshold=0.25) # this function reads the best model from the experiment folder and generates the submission file