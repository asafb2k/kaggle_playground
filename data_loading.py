import pandas as pd
import numpy as np
import os
import tifffile
import torch
from torch.utils.data import Dataset
import scipy


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
        

class Compose:
    """Composes several transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, volume):
        for transform in self.transforms:
            volume = transform(volume)
        return volume


class RandomNoise:
    """Add random Gaussian noise to the volume."""
    def __init__(self, std=0.02, p=0.5):
        self.std = std
        self.p = p
        
    def __call__(self, volume):
        if np.random.rand() < self.p:
            noise = np.random.normal(0, self.std, volume.shape).astype(volume.dtype)
            volume = volume + noise
            # Clip to ensure values stay within valid range
            volume = np.clip(volume, 0, 1)
        return volume

class Normalize:
    """Normalize the volume using mean and std."""
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std
        
    def __call__(self, volume):
        return (volume - self.mean) / self.std

# Create training and validation transforms
def get_train_transform(mean=0.5, std=0.5):
    return Compose([
        RandomNoise(std=0.02, p=0.5),
        Normalize(mean=mean, std=std)
    ])

def get_val_transform(mean=0.5, std=0.5):
    return Compose([
        Normalize(mean=mean, std=std)
    ])

# Utility function to calculate dataset mean and std
def calculate_dataset_stats(dataloader):
    """Calculate mean and std of the dataset for normalization."""
    mean_sum = 0
    var_sum = 0
    count = 0
    
    for inputs, _ in dataloader:
        batch_size = inputs.size(0)
        inputs = inputs.view(batch_size, -1)
        mean_sum += inputs.mean(1).sum(0)
        var_sum += inputs.var(1).sum(0)
        count += batch_size
    
    mean = mean_sum / count
    std = torch.sqrt(var_sum / count)
    
    return mean.item(), std.item()
