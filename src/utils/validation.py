import torch

"""
This is a Python module that contains helper functions for validating input arguments.
The plan is to collect them here and then create a universal validation decorator @validate_args 
to check all the input arguments against the expected types specified e.g.
as class attributes.
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
