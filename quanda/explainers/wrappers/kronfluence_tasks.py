"""Reusable `kronfluence.task.Task` subclasses for quanda benchmarks."""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from kronfluence.task import Task  # type: ignore
from torch import nn


class TextClassificationTask(Task):
    """HuggingFace-style text-classification task for Kronfluence.

    Mirrors the `glue` example in kronfluence:
    https://github.com/pomonam/kronfluence/tree/main/examples/glue.

    Parameters
    ----------
    tracked_modules : Optional[List[str]]
        Names of modules to restrict influence computation to (as in
        `model.named_modules()`). If None, kronfluence tracks every
        applicable `nn.Linear` / `nn.Conv2d`.

    """

    def __init__(self, tracked_modules: Optional[List[str]] = None) -> None:
        """Initialize the task."""
        self._tracked_modules = tracked_modules

    def compute_train_loss(
        self,
        batch: Dict[str, torch.Tensor],
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        """Compute the training loss for the given batch and model."""
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        )

        if not sample:
            return F.cross_entropy(logits, batch["labels"], reduction="sum")
        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
            sampled_labels = torch.multinomial(probs, num_samples=1).flatten()
        return F.cross_entropy(logits, sampled_labels, reduction="sum")

    def compute_measurement(
        self,
        batch: Dict[str, torch.Tensor],
        model: nn.Module,
    ) -> torch.Tensor:
        """Compute the influence measurement for the given batch and model."""
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        )

        labels = batch["labels"]
        bindex = torch.arange(logits.shape[0]).to(
            device=logits.device, non_blocking=False
        )
        logits_correct = logits[bindex, labels]

        cloned_logits = logits.clone()
        cloned_logits[bindex, labels] = torch.tensor(
            -torch.inf, device=logits.device, dtype=logits.dtype
        )

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return -margins.sum()

    def get_attention_mask(
        self, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Return the attention mask for the given batch."""
        return batch["attention_mask"]

    def get_influence_tracked_modules(self) -> Optional[List[str]]:
        """Return the list of modules for influence computation."""
        return self._tracked_modules


__all__ = ["TextClassificationTask"]
