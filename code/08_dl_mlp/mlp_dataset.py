"""
Data loading and PyTorch Dataset creation for the MLP model.
Connects to the shared preprocessing pipeline to ensure consistency
with classical ML experiments.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import os

# Adopt the shared preprocessing logic from the main code directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shared

class StrokeDataset(Dataset):
    """
    PyTorch Dataset for the Stroke Prediction data.
    """
    def __init__(self, X, y):
        """
        Initialize the dataset with preprocessed features and labels.
        Expected shapes: X -> (N, num_features), y -> (N, 1)
        """
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        # Reshape y to (N, 1) to match BCEWithLogitsLoss requirements
        self.y = torch.tensor(np.array(y), dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(batch_size=32, val_size=0.2, random_state=42):
    """
    Loads data using shared.py utilities, applies the preprocessing,
    and returns PyTorch DataLoaders for training, validation, and testing.
    """
    # 1. Load data using shared.py
    df = shared.load_data()
    
    # 2. Get base train/test split (stratified 80/20)
    X_train_full, X_test, y_train_full, y_test = shared.get_split(df)
    
    # 3. Create a dedicated validation split from the train split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, 
        test_size=val_size, 
        stratify=y_train_full, 
        random_state=random_state
    )
    
    # 4. Get and fit preprocessor ONLY on the new training split
    preprocessor = shared.get_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)
    
    # 5. Instantiate StrokeDataset
    train_ds = StrokeDataset(X_train_proc, y_train)
    val_ds = StrokeDataset(X_val_proc, y_val)
    test_ds = StrokeDataset(X_test_proc, y_test)
    
    # 6. Create and return PyTorch DataLoaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    input_dim = X_train_proc.shape[1]
    
    return train_loader, val_loader, test_loader, input_dim
