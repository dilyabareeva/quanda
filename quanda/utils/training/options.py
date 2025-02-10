"""Module containing mappings for trainer options."""

import torch
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import (
    StepLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    OneCycleLR,
    ConstantLR,
)

# Optimizer mappings
optimizers = {
    "sgd": SGD,
    "adam": Adam,
    "adamw": AdamW,
    "rmsprop": RMSprop,
}

# Loss function mappings
criteria = {
    "mse": torch.nn.MSELoss,
    "cross_entropy": torch.nn.CrossEntropyLoss,
    "bce": torch.nn.BCELoss,
    "bce_with_logits": torch.nn.BCEWithLogitsLoss,
    "l1": torch.nn.L1Loss,
    "smooth_l1": torch.nn.SmoothL1Loss,
    "nll": torch.nn.NLLLoss,
}

# Learning rate scheduler mappings
schedulers = {
    "step": StepLR,
    "cosine": CosineAnnealingLR,
    "reduce_on_plateau": ReduceLROnPlateau,
    "one_cycle": OneCycleLR,
    "constant": ConstantLR,
}
