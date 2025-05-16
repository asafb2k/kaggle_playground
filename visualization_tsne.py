import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import os

# Import the dataset classes and model from your original script
from semi_supervized_with_pseudo_labeling import ForamDataset, ForamUnlabeledDataset

class Foram3DCNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(Foram3DCNNFeatureExtractor, self).__init__()
        
        # Convolutional layers (same as original model)
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(128)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        
    def forward(self, x):
        # Apply convolutional layers
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
        
        # Flatten the output to create feature vector
        # Output shape will be [batch_size, 128 * 8 * 8 * 8]
        return x.view(x.size(0), -1)

def load_trained_weights(feature_extractor, full_model_path):
    """Load weights from trained model into feature extractor"""
    # Load the full model state dict
    state_dict = torch.load(full_model_path, map_location='cpu')
    
    # Create a new state dict with only the convolutional layers
    feature_extractor_dict = {}
    for name, param in state_dict.items():
        # Only copy the convolutional layers and batch norm layers
        if any(x in name for x in ['conv', 'bn']):
            feature_extractor_dict[name] = param
    
    # Load the weights into the feature extractor
    feature_extractor.load_state_dict(feature_extractor_dict)
    return feature_extractor

def extract_features(model, dataloader, device):
    """Extract features from the data using the model"""
    model.eval()
    features = []
    labels = []
    ids = []
    is_labeled = True  # Flag to track if we're processing labeled data
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            # Handle both labeled and unlabeled data
            if len(batch) == 2:
                inputs, label_or_id = batch
                if isinstance(label_or_id, torch.Tensor) and label_or_id.dtype == torch.long:
                    # This is labeled data
                    curr_labels = label_or_id
                    curr_ids = None
                else:
                    # This is unlabeled data
                    curr_labels = None
                    curr_ids = label_or_id
                    is_labeled = False
            
            inputs = inputs.to(device)
            batch_features = model(inputs)
            
            features.append(batch_features.cpu().numpy())
            if curr_labels is not None:
                labels.append(curr_labels.numpy())
            if curr_ids is not None:
                ids.extend(curr_ids.tolist())
    
    features = np.concatenate(features, axis=0)
    
    if is_labeled:
        labels = np.concatenate(labels, axis=0)
        return features, labels
    else:
        return features, ids

def plot_3d_tsne(features, labels, save_path, title="t-SNE Visualization of CNN Features", show_only_specific_classes=None):
    """Create and save 3D t-SNE plot"""
    # Perform t-SNE
    print("Performing t-SNE reduction...")
    tsne = TSNE(n_components=3, random_state=42, perplexity=30)
    features_tsne = tsne.fit_transform(features)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # If we have labels (for labeled data)
    if isinstance(labels, np.ndarray):
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if show_only_specific_classes is not None and label in show_only_specific_classes:
                mask = labels == label
                ax.scatter(
                    features_tsne[mask, 0],
                    features_tsne[mask, 1],
                    features_tsne[mask, 2],
                    c=[color],
                    label=f'Class {label}',
                    alpha=0.6
                )
        plt.legend()
    else:
        # For unlabeled data, use a single color
        ax.scatter(
            features_tsne[:, 0],
            features_tsne[:, 1],
            features_tsne[:, 2],
            alpha=0.6
        )
    
    ax.set_title(title)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_zlabel('t-SNE 3')

    # lets show the plot and play with it interactively
    plt.show()
    plt.pause(500)
    # Save the plot
    # plt.savefig(save_path)
    # plt.close()
    # print(f"Plot saved to {save_path}")

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Find the latest experiment folder and best model
    experiments_directory = Path("experiments")
    experiment_folders = list(experiments_directory.glob("semi_supervised_learning_*"))
    
    if not experiment_folders:
        raise ValueError("No experiment folders found!")
    
    latest_experiment = max(experiment_folders, key=lambda x: os.path.getmtime(x))
    print(f"Using latest experiment: {latest_experiment}")
    
    # Find the best model checkpoint
    model_files = list(latest_experiment.glob('best_model_acc_val_epoch_*.pth'))
    if not model_files:
        raise ValueError(f"No model checkpoints found in {latest_experiment}")
    
    best_model_path = max(model_files, key=lambda x: float(str(x).split('_')[-1].replace('.pth', '')))
    print(f"Using best model: {best_model_path}")
    
    # Initialize feature extractor and load weights
    feature_extractor = Foram3DCNNFeatureExtractor()
    feature_extractor = load_trained_weights(feature_extractor, best_model_path)
    feature_extractor = feature_extractor.to(device)
    
    # Load labeled data
    labeled_data_path = Path(r'data/labelled.csv')
    labeled_volume_dir = Path(r'data/volumes/volumes/labelled')
    
    labeled_df = pd.read_csv(labeled_data_path)
    labeled_dataset = ForamDataset(labeled_df, labeled_volume_dir)
    labeled_loader = DataLoader(labeled_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Load unlabeled data
    unlabeled_data_path = Path(r'data/unlabelled.csv')
    unlabeled_volume_dir = Path(r'data/volumes/volumes/unlabelled')
    
    unlabeled_df = pd.read_csv(unlabeled_data_path)
    unlabeled_dataset = ForamUnlabeledDataset(unlabeled_df, unlabeled_volume_dir)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Create visualization directory
    vis_dir = latest_experiment / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    # Extract and visualize features for labeled data
    print("\nProcessing labeled data...")
    labeled_features, labeled_labels = extract_features(feature_extractor, labeled_loader, device)
    plot_3d_tsne(
        labeled_features,
        labeled_labels,
        vis_dir / 'tsne_labeled_data.png',
        "t-SNE Visualization of CNN Features (Labeled Data)"
    )
    
    # Extract and visualize features for unlabeled data
    print("\nProcessing unlabeled data...")
    unlabeled_features, unlabeled_ids = extract_features(feature_extractor, unlabeled_loader, device)
    plot_3d_tsne(
        unlabeled_features,
        unlabeled_ids,
        vis_dir / 'tsne_unlabeled_data.png',
        "t-SNE Visualization of CNN Features (Unlabeled Data)"
    )
    
    # Save the features for potential future use
    np.save(vis_dir / 'labeled_features.npy', labeled_features)
    np.save(vis_dir / 'labeled_labels.npy', labeled_labels)
    np.save(vis_dir / 'unlabeled_features.npy', unlabeled_features)
    np.save(vis_dir / 'unlabeled_ids.npy', unlabeled_ids)
    
    print(f"\nVisualization and features saved in {vis_dir}")

if __name__ == "__main__":
    main() 