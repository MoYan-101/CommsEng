"""
torch_losses.py

Implements PyTorch-specific loss functions (MSE, MAE, etc.).
"""

import torch.nn.functional as F

def mse_loss(pred, target):
    """
    Mean Squared Error.
    """
    return F.mse_loss(pred, target)

def mae_loss(pred, target):
    """
    Mean Absolute Error.
    """
    return F.l1_loss(pred, target)

def get_torch_loss_fn(name="mse"):
    """
    Return the specified PyTorch loss function by name.
    :param name: "mse" or "mae"
    :return: callable that takes (pred, target) -> scalar loss
    """
    if name.lower() == "mse":
        return mse_loss
    elif name.lower() == "mae":
        return mae_loss
    else:
        raise ValueError(f"Unknown loss: {name}")
