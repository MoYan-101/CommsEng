"""
data_preprocessing/my_dataset.py

Defines a custom PyTorch Dataset to facilitate DataLoader usage
when training with PyTorch-based models (like the ANN).
"""

import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    """
    A simple Dataset for multi-output regression using PyTorch.
    """
    def __init__(self, X, Y):
        """
        Constructor for MyDataset.
        :param X: NumPy array of input features
        :param Y: NumPy array of output targets
        """
        # Convert to torch.float32 Tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self):
        """
        Return the total number of samples.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Return a single sample (x, y) at index idx.
        """
        return self.X[idx], self.Y[idx]
