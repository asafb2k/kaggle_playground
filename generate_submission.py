import os
import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from Foram3DCNN import Foram3DCNN
from pathlib import Path

# Dataset class for loading unlabeled 3D volumes
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
                # Format: foram_[5 digit id]_sc_[scale]
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

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load labeled data
    unlabeled_csv_path = Path(r'data\unlabelled.csv')  # Update with your path
    unlabeled_volume_dir = Path(r'data\volumes\volumes\unlabelled')  # Update with your path
    
    unlabeled_df = pd.read_csv(unlabeled_csv_path)
    print(f"Loaded {len(unlabeled_df)} unlabeled samples")
    
    # Create dataset and dataloader for unlabeled data
    unlabeled_dataset = ForamUnlabeledDataset(unlabeled_df, unlabeled_volume_dir)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    # Initialize the model with 14 classes (same as the training model)
    model = Foram3DCNN(num_classes=14)
    
    # Load the checkpoint
    checkpoint_path = Path(r"experiments\experiment_1\best_model_acc_val_0.40540540540540543.pth").as_posix()  # Update with your checkpoint path
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    
    # Generate predictions
    all_ids = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, ids in tqdm(unlabeled_loader, desc="Generating predictions"):
            inputs = inputs.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get probabilities
            probs = F.softmax(outputs, dim=1)
            
            # Calculate confidence (max probability)
            max_probs, preds = torch.max(probs, 1)
            
            # Apply threshold for unknown class (class 14)
            # If confidence is below threshold, assign to unknown class
            threshold = 0.4  # Adjust this threshold based on validation results
            unknown_mask = max_probs < threshold
            final_preds = preds.clone()
            final_preds[unknown_mask] = 14  # Set low confidence predictions to unknown class
            
            # Store results
            all_ids.extend(ids.cpu().numpy().astype(int))
            all_preds.extend(final_preds.cpu().numpy())
    
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': all_ids,
        'label': all_preds
    })
    
    # Sort by ID to match the expected order
    submission_df = submission_df.sort_values('id')
    
    # Save submission file
    submission_path = 'submission.csv'
    submission_df.to_csv(submission_path, index=False)
    
    print(f"Submission file saved to {submission_path}")
    
    # Print prediction statistics
    print("\nClass distribution in predictions:")
    value_counts = submission_df['label'].value_counts().sort_index()
    for class_id, count in value_counts.items():
        print(f"Class {class_id}: {count} samples ({count/len(submission_df)*100:.2f}%)")

if __name__ == "__main__":
    main()