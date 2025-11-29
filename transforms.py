import torch
import numpy as np
import random

class Compose:
    """Composes several transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, volume):
        for transform in self.transforms:
            volume = transform(volume)
        return volume

class RandomFlip3D:
    """Randomly flip the volume along x, y, and z axes."""
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, volume):
        # volume shape: (C, D, H, W) or (D, H, W)
        # We assume input is numpy array from tifffile, typically (D, H, W)
        # If it has channel dim, we need to handle it.
        # Based on data_loading.py, volume is loaded as (D, H, W) then reshaped to (1, D, H, W)
        # But transforms are applied BEFORE converting to tensor in data_loading.py?
        # Let's check data_loading.py again.
        # In data_loading.py:
        # volume = volume.reshape(1, *volume.shape) -> (1, D, H, W)
        # volume = volume.astype(np.float32) / volume.max()
        # if self.transform: volume = self.transform(volume)
        
        # So input is (1, D, H, W) numpy array.
        
        if random.random() < self.p:
            # Flip depth
            volume = np.flip(volume, axis=1).copy()
        if random.random() < self.p:
            # Flip height
            volume = np.flip(volume, axis=2).copy()
        if random.random() < self.p:
            # Flip width
            volume = np.flip(volume, axis=3).copy()
            
        return volume

class RandomRotate90_3D:
    """Randomly rotate the volume by 90 degrees along axes."""
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, volume):
        if random.random() < self.p:
            # Pick an axis to rotate around: 1(D), 2(H), 3(W)
            # axes=(axis1, axis2)
            k = random.randint(1, 3) # number of 90 degree rotations
            axis = random.choice([(1, 2), (1, 3), (2, 3)])
            volume = np.rot90(volume, k=k, axes=axis).copy()
        return volume

class RandomIntensityScale:
    """Randomly scale the intensity of the volume."""
    def __init__(self, scale_limit=0.1, p=0.5):
        self.scale_limit = scale_limit
        self.p = p
        
    def __call__(self, volume):
        if random.random() < self.p:
            scale = 1.0 + random.uniform(-self.scale_limit, self.scale_limit)
            volume = volume * scale
            volume = np.clip(volume, 0, 1)
        return volume

class RandomNoise:
    """Add random Gaussian noise to the volume."""
    def __init__(self, std=0.02, p=0.5):
        self.std = std
        self.p = p
        
    def __call__(self, volume):
        if random.random() < self.p:
            noise = np.random.normal(0, self.std, volume.shape).astype(volume.dtype)
            volume = volume + noise
            volume = np.clip(volume, 0, 1)
        return volume

class RandomCutout3D:
    """Randomly cut out a chunk of the volume."""
    def __init__(self, length=32, p=0.5):
        self.length = length
        self.p = p
        
    def __call__(self, volume):
        if random.random() < self.p:
            # volume shape: (C, D, H, W)
            c, d, h, w = volume.shape
            
            # Random center
            cd = random.randint(0, d)
            ch = random.randint(0, h)
            cw = random.randint(0, w)
            
            d1 = np.clip(cd - self.length // 2, 0, d)
            d2 = np.clip(cd + self.length // 2, 0, d)
            h1 = np.clip(ch - self.length // 2, 0, h)
            h2 = np.clip(ch + self.length // 2, 0, h)
            w1 = np.clip(cw - self.length // 2, 0, w)
            w2 = np.clip(cw + self.length // 2, 0, w)
            
            volume[:, d1:d2, h1:h2, w1:w2] = 0.0
            
        return volume

class Normalize:
    """Normalize the volume using mean and std."""
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std
        
    def __call__(self, volume):
        return (volume - self.mean) / self.std

def get_train_transform(mean=0.5, std=0.5, mode='all'):
    transforms = []
    
    if mode == 'all':
        transforms.extend([
            RandomFlip3D(p=0.5),
            RandomRotate90_3D(p=0.5),
            RandomIntensityScale(scale_limit=0.1, p=0.5),
            RandomNoise(std=0.02, p=0.5),
        ])
    elif mode == 'flip':
        transforms.append(RandomFlip3D(p=0.5))
    elif mode == 'none':
        pass
        
    transforms.append(Normalize(mean=mean, std=std))
    
    return Compose(transforms)

def get_val_transform(mean=0.5, std=0.5):
    return Compose([
        Normalize(mean=mean, std=std)
    ])
