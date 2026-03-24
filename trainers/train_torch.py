"""
Trains a PyTorch model with DataLoader, optional Early Stopping, etc.
"""

import torch
import numpy as np
import copy
from torch.utils.data import DataLoader

def train_torch_model_dataloader(model,
                                 train_dataset,
                                 val_dataset,
                                 loss_fn,
                                 epochs=30,
                                 batch_size=32,
                                 lr=1e-3,
                                 weight_decay=0.0,
                                 checkpoint_path=None,
                                 log_interval=5,
                                 early_stopping=False,
                                 patience=5,
                                 optimizer_name="Adam"):
    """
    Train a PyTorch model using DataLoader, with optional Early Stopping & Dropout.
    Training-set metrics are computed in eval mode so they are not affected by
    Dropout or other training-time regularization.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer_name = optimizer_name.strip().lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        print(f"[INFO] Using AdamW optimizer (lr={lr}, weight_decay={weight_decay})")
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        print(f"[INFO] Using Adam optimizer (lr={lr}, weight_decay={weight_decay})")
    else:
        raise ValueError(f"Unknown optimizer_name='{optimizer_name}', only support 'Adam' or 'AdamW' so far.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    # Store per-epoch train/validation metrics, both computed in eval mode.
    train_losses = []
    val_losses = []

    no_improve_epochs = 0

    for epoch in range(1, epochs + 1):
        # ---- Training (for parameter updates only) ----
        model.train()
        for (X_batch, Y_batch) in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            pred_batch = model(X_batch)
            loss_batch = loss_fn(pred_batch, Y_batch)

            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

        # ---- Compute "clean" training metrics with dropout and similar regularization disabled ----
        model.eval()
        train_loss_list = []
        with torch.no_grad():
            for (X_batch, Y_batch) in train_loader:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                pred_batch = model(X_batch)
                loss_batch = loss_fn(pred_batch, Y_batch)
                train_loss_list.append(loss_batch.item())
        train_loss = float(np.mean(train_loss_list))

        # ---- Validation (already in eval mode) ----
        val_loss_list = []
        with torch.no_grad():
            for (X_val_b, Y_val_b) in val_loader:
                X_val_b = X_val_b.to(device)
                Y_val_b = Y_val_b.to(device)
                val_pred_b = model(X_val_b)
                loss_b = loss_fn(val_pred_b, Y_val_b)
                val_loss_list.append(loss_b.item())
        val_loss = float(np.mean(val_loss_list))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % log_interval == 0:
            print(f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve_epochs = 0
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
        else:
            no_improve_epochs += 1

        if early_stopping and no_improve_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch}. Best val_loss={best_val_loss:.6f} at epoch {best_epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model with val_loss={best_val_loss:.6f} from epoch {best_epoch}.")

    return model, train_losses, val_losses
