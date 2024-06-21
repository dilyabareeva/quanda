import torch

"""
This module contains utility functions for validation. The plan is to
move the validation logic into a validation decorator at a later point.
"""


def validate_1d_tensor_or_int_list(targets):
    if isinstance(targets, torch.Tensor):
        if len(targets.shape) != 1:
            raise ValueError(f"targets should be a 1D tensor. Got shape {targets.shape} instead.")
    elif isinstance(targets, list):
        if not all(isinstance(x, int) for x in targets):
            raise ValueError(f"targets should be a list of integers. Got {targets} instead.")
    else:
        raise TypeError(f"targets should be of type List or torch.Tensor. Got {type(targets)} instead.")
