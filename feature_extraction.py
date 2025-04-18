import numpy as np
from pathlib import Path

def extract_features(volume):
    """Extract handcrafted features from a 3D volume"""
    features = []
    
    # Basic statistics
    features.append(np.mean(volume))
    features.append(np.std(volume))
    features.append(np.percentile(volume, 25))
    features.append(np.percentile(volume, 50))
    features.append(np.percentile(volume, 75))
    
    # Shape features
    # Create binary volume
    threshold = np.mean(volume)
    binary = volume > threshold
    # Count voxels
    features.append(np.sum(binary))
    
    # Spatial features
    # Get coordinates of foreground voxels
    coords = np.where(binary)
    if np.sum(binary) > 0:
        # Calculate centroid
        centroid = [np.mean(coords[0]), np.mean(coords[1]), np.mean(coords[2])]
        features.extend(centroid)
        
        # Calculate bounding box dimensions
        x_min, x_max = np.min(coords[0]), np.max(coords[0])
        y_min, y_max = np.min(coords[1]), np.max(coords[1])
        z_min, z_max = np.min(coords[2]), np.max(coords[2])
        features.append(x_max - x_min)
        features.append(y_max - y_min)
        features.append(z_max - z_min)
    else:
        features.extend([64, 64, 64, 0, 0, 0])  # Default values
    
    return np.array(features)



if __name__ == "__main__":
    DATA_DIR = Path('data')
    labelled_path = DATA_DIR / 'volumes/volumes/labelled/'
    # Example usage
    volume = np.random.rand(64, 64, 64)  # Replace with actual volume data
    features = extract_features(volume)
    print("Extracted features:", features)