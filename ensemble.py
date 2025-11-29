import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from pathlib import Path
import tifffile
from sklearn.metrics import classification_report, accuracy_score

# Import custom modules
from models import resnet10_3d, resnet18_3d, resnet34_3d
from transforms import Compose, Normalize

def get_val_transform(mean=0.15, std=0.25):
    return Compose([
        Normalize(mean=mean, std=std)
    ])

class ForamDataset(torch.utils.data.Dataset):
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

def load_model(model_path, model_type='resnet18', num_classes=15, device='cuda'):
    if model_type == 'resnet18':
        model = resnet18_3d(num_classes=num_classes)
    elif model_type == 'resnet10':
        model = resnet10_3d(num_classes=num_classes)
    elif model_type == 'resnet34':
        model = resnet34_3d(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model = model.to(device)
    model.eval()
    return model

def get_predictions(model, dataloader, device):
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Inference"):
            inputs = inputs.to(device)
            
            # TTA
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            
            # Flip depth
            probs += F.softmax(model(torch.flip(inputs, [2])), dim=1)
            # Flip height
            probs += F.softmax(model(torch.flip(inputs, [3])), dim=1)
            # Flip width
            probs += F.softmax(model(torch.flip(inputs, [4])), dim=1)
            
            probs /= 4.0
            
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    return np.concatenate(all_probs, axis=0), np.array(all_labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True, help='Paths to model checkpoints')
    parser.add_argument('--types', nargs='+', required=True, help='Model types (resnet10 or resnet18)')
    parser.add_argument('--weights', nargs='+', type=float, help='Weights for each model')
    args = parser.parse_args()
    
    if len(args.models) != len(args.types):
        raise ValueError("Number of models and types must match")
        
    if args.weights is None:
        args.weights = [1.0] * len(args.models)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data Loading (Validation Set)
    # We need to recreate the exact validation split
    labeled_df = pd.read_csv(r'data\labelled.csv')
    labeled_volume_dir = Path(r'data\volumes\volumes\labelled')
    
    from sklearn.model_selection import train_test_split
    _, val_df = train_test_split(labeled_df, test_size=0.5, stratify=labeled_df['label'], random_state=42)
    
    val_dataset = ForamDataset(val_df, labeled_volume_dir, transform=get_val_transform())
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    # Get predictions from each model
    ensemble_probs = np.zeros((len(val_df), 15))
    total_weight = sum(args.weights)
    
    true_labels = None
    
    for i, (model_path, model_type, weight) in enumerate(zip(args.models, args.types, args.weights)):
        print(f"\nProcessing model {i+1}/{len(args.models)}: {model_path}")
        model = load_model(model_path, model_type, num_classes=15, device=device)
        
        probs, labels = get_predictions(model, val_loader, device)
        
        if true_labels is None:
            true_labels = labels
        
        # Individual accuracy
        preds = np.argmax(probs, axis=1)
        acc = accuracy_score(true_labels, preds)
        print(f"Model {i+1} Accuracy: {acc:.4f}")
        
        ensemble_probs += probs * weight
        
    ensemble_probs /= total_weight
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    
    ensemble_acc = accuracy_score(true_labels, ensemble_preds)
    print(f"\nEnsemble Accuracy: {ensemble_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(true_labels, ensemble_preds))

if __name__ == '__main__':
    main()
