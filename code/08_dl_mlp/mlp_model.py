"""
PyTorch Model definition for the deep learning pipeline.
"""
import torch
import torch.nn as nn

class StrokeMLP(nn.Module):
    """
    Multi-Layer Perceptron for stroke prediction.
    """
    def __init__(self, input_dim, hidden_dim1=64, hidden_dim2=32, dropout_rate=0.2):
        super().__init__()
        
        # Simple Tabular MLP for binary classification
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim2, 1)  # Single raw logit output
        )

    def forward(self, x):
        """
        Forward pass of the MLP.
        Outputs raw logits for BCEWithLogitsLoss.
        """
        return self.network(x)
