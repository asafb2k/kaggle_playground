import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import argparse
from pathlib import Path
import tifffile
import os

from models import resnet18_3d
from transforms import Compose, Normalize

def get_val_transform(mean=0.15, std=0.25):
    return Compose([
        Normalize(mean=mean, std=std)
    ])

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
        
        if file_path is None:
            # Handle missing file if any (though shouldn't happen based on previous runs)
            print(f"Warning: File for ID {file_id} not found.")
            return torch.zeros((1, 128, 128, 128)), file_id

        volume = tifffile.imread(file_path)
        volume = volume.reshape(1, *volume.shape)
        volume = volume.astype(np.float32) / volume.max()
        
        if self.transform:
            volume = self.transform(volume)
            
        return torch.tensor(volume, dtype=torch.float32), file_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to best model checkpoint')
    parser.add_argument('--output', type=str, default='submission.csv', help='Output submission file')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data Loading
    unlabelled_df = pd.read_csv(r'data\unlabelled.csv')
    unlabeled_volume_dir = Path(r'data\volumes\volumes\unlabelled')
    
    dataset = UnlabeledDataset(unlabelled_df, unlabeled_volume_dir, transform=get_val_transform())
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Load Model
    print(f"Loading model from {args.model_path}")
    model = resnet18_3d(num_classes=15).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    predictions = []
    ids = []
    
    print("Generating predictions...")
    with torch.no_grad():
        for inputs, batch_ids in tqdm(dataloader, desc="Inference"):
            inputs = inputs.to(device)
            
            # TTA
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            
            probs += F.softmax(model(torch.flip(inputs, [2])), dim=1)
            probs += F.softmax(model(torch.flip(inputs, [3])), dim=1)
            probs += F.softmax(model(torch.flip(inputs, [4])), dim=1)
            probs /= 4.0
            
            preds = torch.argmax(probs, dim=1).cpu().numpy()
            
            predictions.extend(preds)
            ids.extend(batch_ids.numpy())
            
    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'id': ids,
        'label': predictions
    })
    
    # Sort by ID to match sample submission order if needed (usually good practice)
    submission_df = submission_df.sort_values('id')
    
    # Save
    submission_df.to_csv(args.output, index=False)
    print(f"Submission saved to {args.output}")
    print(submission_df.head())

if __name__ == '__main__':
    main()
