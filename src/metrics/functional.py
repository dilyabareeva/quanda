"""

WORK IN PROGRESS!!!
"""
import warnings
from typing import Optional, Union

import torch

from src.utils.explanations import (
    BatchedCachedExplanations,
    TensorExplanations,
)
from utils.cache import ExplanationsCache as EC


def function_example(
    model: torch.nn.Module,
    train_dataset: torch.utils.data.Dataset,
    top_k: int = 1,
    explanations: Union[str, torch.Tensor, TensorExplanations, BatchedCachedExplanations] = "./",
    batch_size: Optional[int] = 8,
    device="cpu",
    **kwargs,
):
    """
    I've copied the existing code from the memory-less metric version here, that can be reused in the future here.
    It will not be called "function_example" in the future. There will be many reusable functions, but every metric
    will get a functional version here.

    :param model:
    :param train_dataset:
    :param top_k:
    :param explanations:
    :param batch_size:
    :param device:
    :param kwargs:
    :return:
    """
    if isinstance(explanations, str):
        explanations = EC.load(path=explanations, device=device)
        if explanations.batch_size != batch_size:
            warnings.warn(
                "Batch size mismatch between loaded explanations and passed batch size. The inferred batch "
                "size will be used instead."
            )
            batch_size = explanations[0]
    elif isinstance(explanations, torch.Tensor):
        explanations = TensorExplanations(explanations, batch_size=batch_size, device=device)
