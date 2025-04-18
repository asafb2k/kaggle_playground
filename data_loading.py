import pandas as pd
import numpy as np
import os
import tifffile
import torch
from torch.utils.data import Dataset


class ForamDataset(Dataset):
    def __init__(self, csv_file, volume_dir, transform=None, is_labeled=True):
        self.data_frame = pd.read_csv(csv_file)
        self.volume_dir = volume_dir
        self.transform = transform
        self.is_labeled = is_labeled
        
        # Pre-find all file paths to speed up loading
        self.file_paths = {}
        prefix = "labelled_foram_" if self.is_labeled else "foram_"
        
        for file in os.listdir(self.volume_dir):
            if file.startswith(prefix):
                file_id = int(file.split('_')[2 if self.is_labeled else 1])
                self.file_paths[file_id] = os.path.join(self.volume_dir, file)
        
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        file_id = self.data_frame.iloc[idx, 0]
        
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
        
        if self.is_labeled:
            label = self.data_frame.iloc[idx, 1]
            return volume_tensor, label
        else:
            return volume_tensor, file_id