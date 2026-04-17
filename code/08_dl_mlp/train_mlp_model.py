"""
Training and evaluation loops for the PyTorch MLP model.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    roc_auc_score, average_precision_score
)

# Adopt the shared preprocessing logic for plot styling
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import shared
except ImportError:
    pass

from mlp_model import StrokeMLP
from mlp_dataset import get_dataloaders
from losses import FocalLoss

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
    return running_loss / len(dataloader.dataset)

def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on a validation or test set."""
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            
            # Convert logits to probabilities
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()
            
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
    epoch_loss = running_loss / len(dataloader.dataset)
    
    # Sklearn metrics matching shared.py classical evaluations
    y_true = np.array(all_targets).flatten()
    y_prob = np.array(all_probs).flatten()
    y_pred = np.array(all_preds).flatten()
    
    metrics = {
        "loss": epoch_loss,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob)
    }
    
    return metrics

def plot_history(train_losses, val_losses, loss_type, model_name):
    """Plots and saves the training history."""
    if 'shared' in sys.modules and hasattr(shared, 'setup_plot_style'):
        shared.setup_plot_style()
        
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss", linewidth=2)
    plt.plot(val_losses, label="Val Loss", linewidth=2)
    plt.title(f"Training History ({model_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    os.makedirs("outputs/figures", exist_ok=True)
    plt.savefig(f"outputs/figures/dl_mlp_training_history_{loss_type}.png")
    plt.close()

def main():
    """Main orchestration function for PyTorch MLP training."""
    batch_size = 64
    epochs = 100
    learning_rate = 1e-3
    patience = 10
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Preparing data...")
    train_loader, val_loader, test_loader, input_dim = get_dataloaders(batch_size=batch_size)
    
    # Calculate pos_weight ONCE from the DL training labels only
    train_y = train_loader.dataset.y
    num_pos = train_y.sum().item()
    num_neg = len(train_y) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
    print(f"Calculated pos_weight for imbalanced training: {pos_weight.item():.4f}")
    
    loss_configs = ["bce", "weighted", "focal"]
    all_results = []
    
    os.makedirs("outputs/dl_results", exist_ok=True)
    os.makedirs("outputs/results/dl_mlp", exist_ok=True)
    
    for loss_type in loss_configs:
        print(f"\n{'='*40}")
        print(f"Starting Training for loss type: '{loss_type}'")
        print(f"{'='*40}")
        
        model = StrokeMLP(input_dim=input_dim).to(device)
        best_model_path = f"outputs/dl_results/best_mlp_{loss_type}.pth"
        
        if loss_type == "bce":
            criterion = nn.BCEWithLogitsLoss()
            model_name = "PyTorch MLP (BCE)"
        elif loss_type == "weighted":
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            model_name = "PyTorch MLP (Weighted)"
        elif loss_type == "focal":
            criterion = FocalLoss(gamma=2.0)
            model_name = "PyTorch MLP (Focal)"
        else:
            raise ValueError(f"Unknown loss type {loss_type}")
            
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        best_val_pr_auc = -1.0
        epochs_no_improve = 0
        
        train_losses = []
        val_losses = []
        val_pr_aucs = []
        
        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, criterion, device)
            
            train_losses.append(train_loss)
            val_losses.append(val_metrics["loss"])
            val_pr_aucs.append(val_metrics["pr_auc"])
            
            # Early stopping on Validation PR-AUC
            val_pr_auc = val_metrics["pr_auc"]
            if val_pr_auc > best_val_pr_auc:
                best_val_pr_auc = val_pr_auc
                epochs_no_improve = 0
                torch.save(model.state_dict(), best_model_path)
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val PR-AUC: {val_pr_auc:.4f} *Best*")
            else:
                epochs_no_improve += 1
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val PR-AUC: {val_pr_auc:.4f}")
                
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs (Best Val PR-AUC: {best_val_pr_auc:.4f}).")
                break
        
        print("Plotting history...")
        plot_history(train_losses, val_losses, loss_type, model_name)
                
        print("Evaluating best model on unseen test set...")
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        test_metrics = evaluate(model, test_loader, criterion, device)
        
        result_row = {
            "Model": model_name,
            "Accuracy": test_metrics["accuracy"],
            "F1": test_metrics["f1"],
            "Recall": test_metrics["recall"],
            "Precision": test_metrics["precision"],
            "AUC-ROC": test_metrics["roc_auc"],
            "AUC-PR": test_metrics["pr_auc"]
        }
        all_results.append(result_row)
        
        print(f"Test Performance ({model_name}):")
        for k, v in test_metrics.items():
            if k != "loss":
                print(f"  {k}: {v:.4f}")

    print(f"\n{'='*40}")
    print("Saving consolidated results to CSV...")
    results_df = pd.DataFrame(all_results)
    
    # Match EXACT classical baseline column order implicitly since dict keys match
    output_csv_path = "outputs/results/dl_mlp/test_results.csv"
    results_df.to_csv(output_csv_path, index=False, float_format="%.4f")
    print(f"Successfully saved test metrics to {output_csv_path}")

if __name__ == "__main__":
    main()
