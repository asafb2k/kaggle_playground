import os
import re
from pathlib import Path
import pandas as pd

experiments_dir = Path('experiments')
results = []

for exp_dir in experiments_dir.iterdir():
    if exp_dir.is_dir() and exp_dir.name.startswith('exp'):
        best_acc = 0.0
        best_epoch = -1
        
        # Find best model file
        for file in exp_dir.glob('best_model_acc_val_epoch_*.pth'):
            # format: best_model_acc_val_epoch_{epoch}_{acc}.pth
            try:
                parts = file.stem.split('_')
                acc = float(parts[-1])
                epoch = int(parts[-2])
                
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch
            except:
                continue
        
        results.append({
            'Experiment': exp_dir.name,
            'Best Accuracy': best_acc,
            'Best Epoch': best_epoch
        })

print(f"{'Experiment':<50} | {'Best Acc':<10} | {'Epoch':<5}")
print("-" * 75)
for r in sorted(results, key=lambda x: x['Experiment']):
    print(f"{r['Experiment']:<50} | {r['Best Accuracy']:<10.4f} | {r['Best Epoch']:<5}")
