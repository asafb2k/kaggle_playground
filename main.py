import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_loading import ForamDataset
from Foram3DCNN import Foram3DCNN
import numpy




def train_model(model, labeled_loader, unlabeled_loader, criterion, optimizer, scheduler, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # First, train on labeled data
        for inputs, labels in tqdm(labeled_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Now, use unlabeled data with pseudo-labeling
        if epoch >= 2:  # Start pseudo-labeling after a few epochs
            model.eval()
            
            with torch.no_grad():
                for inputs, _ in tqdm(unlabeled_loader):
                    inputs = inputs.to(device)
                    
                    # Generate pseudo-labels
                    outputs = model(inputs)
                    max_probs, pseudo_labels = torch.max(F.softmax(outputs, dim=1), dim=1)
                    
                    # Only use confident predictions (threshold = 0.9)
                    mask = max_probs > 0.9
                    
                    if mask.sum() > 0:
                        model.train()
                        
                        # Forward pass with confident pseudo-labels
                        optimizer.zero_grad()
                        outputs_pl = model(inputs[mask])
                        loss_pl = criterion(outputs_pl, pseudo_labels[mask])
                        
                        # Weighted loss for pseudo-labels
                        loss_pl = loss_pl * 0.5  # Lower weight for pseudo-labels
                        
                        loss_pl.backward()
                        optimizer.step()
                        
                        running_loss += loss_pl.item() * inputs[mask].size(0)
            
            model.train()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        scheduler.step()
        
        # Save best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), 'best_foram_model.pth')
    
    return model

def train_ensemble(n_models=5):
    models = []
    
    for i in range(n_models):
        print(f"Training model {i+1}/{n_models}")
        model = Foram3DCNN(num_classes=14)  # 14 known classes (no unknown class yet)
        
        # Training code...
        
        models.append(model)
    
    return models

def predict_with_uncertainty(models, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    
    for model in models:
        model.eval()
        model = model.to(device)
        
        predictions = []
        ids = []
        
        with torch.no_grad():
            for inputs, file_ids in dataloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)
                
                predictions.append(probs.cpu().numpy())
                ids.extend(file_ids.numpy())
        
        all_probs = np.vstack(predictions)
        results.append(all_probs)
    
    # Stack all model predictions
    ensemble_probs = np.stack(results)
    
    # Mean probability across models
    mean_probs = np.mean(ensemble_probs, axis=0)
    
    # Uncertainty: entropy of predicted distribution
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)
    
    # Assign high uncertainty samples to "unknown" class (14)
    final_preds = np.argmax(mean_probs, axis=1)
    final_preds[entropy > 1.0] = 14  # Assign to unknown class
    
    return ids, final_preds


def main():
    # Data preparation
    labeled_dataset = ForamDataset('labelled.csv', 'Volumes/labelled/', is_labeled=True)
    unlabeled_dataset = ForamDataset('unlabelled.csv', 'Volumes/unlabelled/', is_labeled=False)
    
    labeled_loader = DataLoader(labeled_dataset, batch_size=4, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=4, shuffle=True)
    
    # Model training
    model = Foram3DCNN(num_classes=15)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    model = train_model(model, labeled_loader, unlabeled_loader, criterion, optimizer, scheduler, num_epochs=20)
    
    # Alternative: Train ensemble
    models = train_ensemble(n_models=3)
    
    # Prediction
    test_loader = DataLoader(unlabeled_dataset, batch_size=4, shuffle=False)
    ids, preds = predict_with_uncertainty(models, test_loader)
    
    # Create submission
    submit_df = pd.DataFrame({'id': ids, 'label': preds})
    submit_df.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()