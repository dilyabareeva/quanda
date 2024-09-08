import inspect
from typing import Any, Callable

import pytorch_lightning as pl
import torch

"""
This module contains utility functions for validation. The plan is to
move the validation logic into a validation decorator at a later point.
"""


def validate_checkpoints_load_func(checkpoints_load_func: Callable[..., Any]) -> None:
    signature = inspect.signature(checkpoints_load_func)
    parameters = list(signature.parameters.values())

    if len(parameters) < 2:
        raise ValueError(f"checkpoints_load_func must have at least 2 required parameters. Got {len(parameters)}.")

    first_param, second_param = parameters[0], parameters[1]

    if first_param.annotation not in [torch.nn.Module, pl.LightningModule]:
        raise TypeError(
            f"The first parameter of checkpoints_load_func must be of type 'torch.nn.Module'. Got '{first_param.annotation}'."
        )

    if second_param.annotation is not str:
        raise TypeError(
            f"The second parameter of checkpoints_load_func must be of type 'str'. Got '{second_param.annotation}'."
        )
