import os
import subprocess
import sys

# Define experiments
experiments = [
    # 1. Baseline: ResNet10, all augs, default params
    {'name': 'exp01_resnet10_base', 'model': 'resnet10', 'lr': 0.0001, 'aug_mode': 'all', 'epochs': 50},
    
    # 2. No Augmentation: check baseline
    {'name': 'exp02_resnet10_no_aug', 'model': 'resnet10', 'lr': 0.0001, 'aug_mode': 'none', 'epochs': 50},
    
    # 3. Light Augmentation: Flip only
    {'name': 'exp03_resnet10_flip_only', 'model': 'resnet10', 'lr': 0.0001, 'aug_mode': 'flip', 'epochs': 50},
    
    # 4. Deeper Model: ResNet18
    {'name': 'exp04_resnet18_base', 'model': 'resnet18', 'lr': 0.0001, 'aug_mode': 'all', 'epochs': 50},
    
    # 5. Higher Learning Rate
    {'name': 'exp05_resnet10_lr_1e-3', 'model': 'resnet10', 'lr': 0.001, 'aug_mode': 'all', 'epochs': 50},
    
    # 6. Lower Learning Rate
    {'name': 'exp06_resnet10_lr_1e-5', 'model': 'resnet10', 'lr': 0.00001, 'aug_mode': 'all', 'epochs': 50},
    
    # 7. Early Pseudo-labeling
    {'name': 'exp07_resnet10_pseudo_early', 'model': 'resnet10', 'lr': 0.0001, 'aug_mode': 'all', 'epochs': 50, 'pseudo_start': 10},
    
    # 8. Late Pseudo-labeling
    {'name': 'exp08_resnet10_pseudo_late', 'model': 'resnet10', 'lr': 0.0001, 'aug_mode': 'all', 'epochs': 50, 'pseudo_start': 40},
    
    # 9. High Weight Decay
    {'name': 'exp09_resnet10_wd_high', 'model': 'resnet10', 'lr': 0.0001, 'aug_mode': 'all', 'epochs': 50, 'weight_decay': 0.001},
    
    # 10. Small Batch Size
    {'name': 'exp10_resnet10_bs_8', 'model': 'resnet10', 'lr': 0.0001, 'aug_mode': 'all', 'epochs': 50, 'batch_size': 8},
]

# python_executable = sys.executable
# Use the specific conda environment python found
python_executable = r'C:\Users\USER\anaconda3\envs\kaggle_py3911\python.exe'
script_path = 'semi_supervized_with_pseudo_labeling.py'

print(f"Starting {len(experiments)} experiments...")

for i, exp in enumerate(experiments):
    print(f"\nRunning Experiment {i+1}/{len(experiments)}: {exp['name']}")
    
    cmd = [python_executable, script_path]
    
    # Add arguments
    cmd.extend(['--exp_name', exp['name']])
    cmd.extend(['--model', exp['model']])
    cmd.extend(['--lr', str(exp['lr'])])
    cmd.extend(['--aug_mode', exp['aug_mode']])
    cmd.extend(['--epochs', str(exp['epochs'])])
    
    if 'pseudo_start' in exp:
        cmd.extend(['--pseudo_start', str(exp['pseudo_start'])])
    if 'weight_decay' in exp:
        cmd.extend(['--weight_decay', str(exp['weight_decay'])])
    if 'batch_size' in exp:
        cmd.extend(['--batch_size', str(exp['batch_size'])])
        
    print(f"Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Experiment {exp['name']} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Experiment {exp['name']} failed with error: {e}")
        # Decide whether to continue or stop. Let's continue.
        continue

print("\nAll experiments finished.")
