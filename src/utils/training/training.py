from typing import Optional

import torch


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.dataloader.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.modules.loss._Loss,
    device: str = "cpu",
    max_epochs: int = 100,
    val_loader: torch.utils.data.dataloader.DataLoader = None,
    early_stopping: bool = False,
    early_stopping_kwargs: Optional[dict] = {"patience": 10},
    verbose: bool = False,
    *args,
    **kwargs,
):
    """
    Function to train a model.

    Args:
        model: torch.nn.Module: Model to train.
        train_loader: torch.utils.data.dataloader.DataLoader: DataLoader for training data.
        optimizer: torch.optim.Optimizer: Optimizer to use for training.
        criterion: torch.nn.modules.loss._Loss: Loss function to use for training.
        device: str: Device to use for training.
        max_epochs: int: Maximum number of epochs to train for.
        val_loader: torch.utils.data.dataloader.DataLoader: DataLoader for validation data.
        early_stopping: bool: Whether to use early stopping.
        patience: int: Patience for early stopping.
        metric: str: Metric to use for early stopping.
        verbose: bool: Whether to print training information.
        *args: Additional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        model: torch.nn.Module: Trained model.
    """
    model.to(device)
    if early_stopping:
        assert val_loader is not None, "Validation loader is required for early stopping."
        assert "metric" in early_stopping_kwargs, "Metric is required for early stopping."
        assert "patience" in early_stopping_kwargs, "Patience is required for early stopping."
        patience = early_stopping_kwargs["patience"]

    no_improvement = 0
    best_metric = None

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        if train_loss < best_metric:
            best_metric = loss
            no_improvement = 0
        else:
            no_improvement += 1

        if early_stopping and no_improvement >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch}.")
            break

        if verbose:  # TODO: tqdm
            print(f"Epoch: {epoch}, Train Loss: {train_loss}")
