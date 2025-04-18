import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from pathlib import Path
import os
from tqdm import tqdm
import time
import cv2
from data_loading import ForamDataset

import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Define the CNN model for 3D volumes (now with 2 classes for binary classification)
class Foram3DCNNBinary(nn.Module):
    def __init__(self):
        super(Foram3DCNNBinary, self).__init__()
        
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
        self.fc2 = nn.Linear(512, 2)  # Binary classifier (1 for target class, 0 for all others)
        
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

# Unlabeled dataset for inference
class ForamUnlabeledDataset(Dataset):
    def __init__(self, data_frame, volume_dir, transform=None):
        self.data_frame = data_frame
        self.volume_dir = volume_dir
        self.transform = transform
        
        # Pre-find all file paths to speed up loading
        self.file_paths = {}
        prefix = "foram_"
        
        for file in os.listdir(self.volume_dir):
            if file.startswith(prefix):
                # Extract the ID from the filename
                parts = file.split('_')
                file_id = int(parts[1])
                self.file_paths[file_id] = os.path.join(self.volume_dir, file)
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        file_id = self.data_frame.iloc[idx]['id']
        
        # Get pre-found file path
        file_path = self.file_paths.get(file_id)
        if not file_path:
            raise FileNotFoundError(f"Could not find file for ID {file_id}")
        
        # Load volume
        volume = tifffile.imread(file_path)
        
        # Add channel dimension and normalize
        volume = volume.reshape(1, *volume.shape)
        volume = volume.astype(np.float32) / volume.max()  # Normalize to [0,1]
        
        if self.transform:
            volume = self.transform(volume)
        
        volume_tensor = torch.tensor(volume, dtype=torch.float32)
        
        return volume_tensor, file_id

# Training and validation functions
def train_epoch(model, dataloader, criterion, optimizer, device, target_class, scaler=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels, _ in tqdm(dataloader, desc=f"Training Class {target_class}"):
        inputs = inputs.to(device)
        
        # Convert multi-class labels to binary (1 for target class, 0 for others)
        binary_labels = (labels == target_class).long().to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        if scaler:  # Use mixed precision training if scaler is provided
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, binary_labels)
                
            # Scale loss and do backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Regular training
            outputs = model(inputs)
            loss = criterion(outputs, binary_labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += binary_labels.size(0)
        correct += (predicted == binary_labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device, target_class):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # For ROC-AUC calculation
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels, _ in tqdm(dataloader, desc=f"Validating Class {target_class}"):
            inputs = inputs.to(device)
            
            # Convert multi-class labels to binary (1 for target class, 0 for others)
            binary_labels = (labels == target_class).long().to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, binary_labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            # Get probabilities for positive class
            probs = F.softmax(outputs, dim=1)[:, 1]
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += binary_labels.size(0)
            correct += (predicted == binary_labels).sum().item()
            
            # Store probabilities and labels for ROC-AUC
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(binary_labels.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    
    # Calculate ROC-AUC if we have both positive and negative samples
    roc_auc = 0
    if len(set(all_labels)) > 1:
        roc_auc = roc_auc_score(all_labels, all_probs)
    
    return epoch_loss, epoch_acc, roc_auc

def train_single_classifier(dataset, target_class, experiment_path, device):
    # Split data into train and validation sets
    train_indices, val_indices = train_test_split(
        range(len(dataset)), 
        test_size=0.35, 
        stratify=[1 if label == target_class else 0 for _, label, _ in dataset],
        random_state=42
    )
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Initialize model, loss function, and optimizer
    model = Foram3DCNNBinary()
    model = model.to(device)
    
    # Use weighted loss to handle class imbalance
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 14.0]).to(device))  # More weight to positive class
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # Use mixed precision training if available
    use_amp = torch.cuda.is_available()
    scaler = GradScaler() if use_amp else None
    
    # Training loop
    num_epochs = 50
    best_val_auc = 0.0
    
    # For tracking metrics
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    val_aucs = []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs} for Class {target_class}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, target_class, scaler)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device, target_class)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_aucs.append(val_auc)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        
        # Save best model based on AUC (better metric for imbalanced data)
        if val_auc > best_val_auc:
            if 'model_save_path' in locals() and os.path.exists(model_save_path):
                os.remove(model_save_path)
            best_val_auc = val_auc
            model_save_path = experiment_path / f'best_model_class_{target_class}_epoch_{epoch}_auc_{best_val_auc:.4f}.pth'
            torch.save(model.state_dict(), model_save_path.as_posix())
            print(f"Saved best model for class {target_class} with AUC: {best_val_auc:.4f}")
    
    # Plot training and validation metrics for this classifier
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Loss for Class {target_class}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Training Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title(f'Accuracy for Class {target_class}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(val_aucs, label='Validation AUC')
    plt.title(f'ROC-AUC for Class {target_class}')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    
    plt.tight_layout()
    fig_save_path = experiment_path / f'training_metrics_class_{target_class}.png'
    plt.savefig(fig_save_path.as_posix())
    plt.close()
    
    return best_val_auc

def train_all_binary_classifiers():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load labeled data
    data_path = Path(r'data\labelled.csv')  # Update with your path
    volume_dir = Path(r'data\volumes\volumes\labelled')  # Update with your path
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} labeled samples")
    
    # Create dataset
    dataset = ForamDataset(df, volume_dir, transform=None)
    
    # Create experiment directory
    experiments_directory = Path("experiments")
    experiment_name = 'binary_classifiers'
    experiment_path = experiments_directory / experiment_name
    os.makedirs(experiment_path, exist_ok=True)
    
    # Train a separate binary classifier for each class
    results = {}
    
    for target_class in range(14):  # 14 classes
        print(f"\n=== Training binary classifier for class {target_class} ===")
        best_auc = train_single_classifier(dataset, target_class, experiment_path, device)
        results[target_class] = best_auc
    
    # Print summary of results
    print("\n=== Binary Classifiers Training Summary ===")
    for class_id, auc in results.items():
        print(f"Class {class_id}: Best validation AUC = {auc:.4f}")
    
    # Save results to file
    with open(experiment_path / 'results_summary.txt', 'w') as f:
        f.write("Binary Classifiers Training Summary\n")
        f.write("=================================\n\n")
        for class_id, auc in results.items():
            f.write(f"Class {class_id}: Best validation AUC = {auc:.4f}\n")

def generate_predictions_from_binary_classifiers():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load unlabeled data
    unlabeled_csv_path = Path(r'data\unlabelled.csv')
    unlabeled_volume_dir = Path(r'data\volumes\volumes\unlabelled')
    
    unlabeled_df = pd.read_csv(unlabeled_csv_path)
    print(f"Loaded {len(unlabeled_df)} unlabeled samples")
    
    # Create dataset and dataloader for unlabeled data
    unlabeled_dataset = ForamUnlabeledDataset(unlabeled_df, unlabeled_volume_dir)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    # Load all binary classifiers
    binary_models = {}
    experiment_path = Path("experiments/binary_classifiers")
    
    for class_id in range(14):
        # Find the best model for this class
        model_files = list(experiment_path.glob(f'best_model_class_{class_id}_*.pth'))
        if not model_files:
            print(f"Warning: No model found for class {class_id}")
            continue
            
        # Use the most recent model file if multiple exist
        model_path = model_files[0]
        
        # Initialize model and load weights
        model = Foram3DCNNBinary()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        binary_models[class_id] = model
        print(f"Loaded model for class {class_id} from {model_path}")
    
    # Generate predictions with all binary classifiers
    all_probs = np.zeros((len(unlabeled_df), 14))  # Store probabilities for each class
    all_probs_dict = {}
    all_ids = []
    
    with torch.no_grad():
        for inputs, ids in tqdm(unlabeled_loader, desc="Generating predictions"):
            inputs = inputs.to(device)
            all_ids.extend(ids.cpu().numpy().astype(int))
            
            all_probs_dict[int(ids)] = []
            # Get predictions from each binary classifier
            for class_id, model in binary_models.items():
                outputs = model(inputs)
                # Get probability of positive class (class 1)
                probs = F.softmax(outputs, dim=1)[:, 1]
                all_probs_dict[int(ids)].append(float(probs.cpu().numpy()[0]))
                all_probs[len(all_ids) - len(ids):len(all_ids), class_id] = probs.cpu().numpy()
            
                # lets show bar chart of probabilities using all_probs_dict
                # for each id, show the probabilities of all classes
                

            # # Create the figure and canvas
            # fig, ax = plt.subplots(figsize=(10, 5))
            # ax.bar(range(14), all_probs_dict[int(ids)])
            # ax.set_title(f"Probabilities for ID {int(ids)}")
            # ax.set_xlabel("Class")
            # ax.set_ylabel("Probability")
            # ax.set_ylim(0, 1)
            # ax.set_xticks(range(14))

            # # Draw the canvas and convert to image
            # canvas = FigureCanvas(fig)
            # canvas.draw()
            # img = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            # img = img.reshape(canvas.get_width_height()[::-1] + (3,))

            # # Convert RGB (matplotlib) to BGR (OpenCV)
            # img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # # Show image with OpenCV
            # cv2.imshow(f"Probabilities for ID {id}", img_bgr)
            # cv2.waitKey(0)

    
    # Determine final class based on highest probability
    # If all probabilities are below threshold, assign to unknown class (14)
    threshold = 0.4  # Adjust this threshold based on validation results
    max_probs = np.max(all_probs, axis=1)
    final_preds = np.argmax(all_probs, axis=1)
    final_preds[max_probs < threshold] = 14  # Unknown class
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': all_ids,
        'label': final_preds
    })
    
    # Sort by ID to match the expected order
    submission_df = submission_df.sort_values('id')
    
    # Save submission file
    submission_path = 'binary_classifiers_submission.csv'
    submission_df.to_csv(submission_path, index=False)
    
    print(f"Submission file saved to {submission_path}")
    
    # Print prediction statistics
    print("\nClass distribution in predictions:")
    value_counts = submission_df['label'].value_counts().sort_index()
    for class_id, count in value_counts.items():
        print(f"Class {class_id}: {count} samples ({count/len(submission_df)*100:.2f}%)")

def validate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    running_loss = 0.0
    correct = 0
    total = 0

    # Load labeled data
    data_path = Path(r'data\labelled.csv')  # Update with your path
    volume_dir = Path(r'data\volumes\volumes\labelled')  # Update with your path
    
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} labeled samples")
            
    val_dataset = ForamDataset(df, volume_dir, transform=None)
    
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Load all binary classifiers
    binary_models = {}
    experiment_path = Path("experiments/binary_classifiers")

    for class_id in range(14):
        # Find the best model for this class
        model_files = list(experiment_path.glob(f'best_model_class_{class_id}_*.pth'))
        if not model_files:
            print(f"Warning: No model found for class {class_id}")
            continue
            
        # Use the most recent model file if multiple exist
        model_path = model_files[0]
        
        # Initialize model and load weights
        model = Foram3DCNNBinary()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        binary_models[class_id] = model
        print(f"Loaded model for class {class_id} from {model_path}")
    
    # For confusion matrix
    all_preds = []
    all_labels = []

    all_probs_dict = {}
    
    
    with torch.no_grad():
        for index ,inputs_labels in tqdm(enumerate(val_loader), desc="Validation"):
            inputs = inputs_labels[0].to(device)
            labels = inputs_labels[1].to(device)
            all_probs_dict[index] = []
            
            for class_id, model in binary_models.items():
                outputs = model(inputs)
                # Get probability of positive class (class 1)
                probs = F.softmax(outputs, dim=1)[:, 1]
                all_probs_dict[index].append(float(probs.cpu().numpy()[0]))
            
            # Calculate accuracy
            predicted = torch.argmax(torch.Tensor(all_probs_dict[index]))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_acc = correct / total
    
    return epoch_acc, all_preds, all_labels

if __name__ == "__main__":
    # First, train all binary classifiers
    # train_all_binary_classifiers()

    # validate()
    
    # Then generate predictions using the trained classifiers
    generate_predictions_from_binary_classifiers()