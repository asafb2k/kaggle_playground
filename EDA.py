import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile
from skimage import io, exposure
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

DATA_DIR = Path('data')
labelled_path = DATA_DIR / 'volumes/volumes/labelled/'


# Load the labeled data
labeled_df = pd.read_csv((DATA_DIR / 'labelled.csv').as_posix())
print(f"Labeled data shape: {labeled_df.shape}")
print(labeled_df.head())

# Count samples per class
class_counts = labeled_df['label'].value_counts().sort_index()
print("\nSamples per class:")
print(class_counts)

# Visualize one example per class
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
axes = axes.flatten()

for class_idx in range(14):
    # Get first image of this class
    sample_idx = int(labeled_df[labeled_df['label'] == class_idx].iloc[0]['id'].split('_')[1])
    # Find the corresponding file
    file_path = None
    for file_path in labelled_path.iterdir():
        if f'labelled_foram_{sample_idx:05d}' in file_path.name:
            break
    
    if file_path:
        # Load and visualize middle slice
        volume = tifffile.imread(file_path.as_posix())
        middle_slice = volume[volume.shape[0]//2, :, :]
        
        axes[class_idx].imshow(middle_slice, cmap='gray')
        axes[class_idx].set_title(f"Class {class_idx}")
        axes[class_idx].axis('off')

plt.tight_layout()
plt.show()

# # Load one sample unlabeled data to understand format
# unlabeled_df = pd.read_csv((DATA_DIR / 'unlabelled.csv').as_posix())
# print(f"\nUnlabeled data shape: {unlabeled_df.shape}")
# print(unlabeled_df.head())