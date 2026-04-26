"""Reusable `kronfluence.task.Task` subclasses for quanda benchmarks."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from kronfluence.task import Task  # type: ignore
from torch import nn


class ImageClassificationTask(Task):
    """Positional image-classification task for Kronfluence.

    Mirrors the `cifar` example in kronfluence:
    https://github.com/pomonam/kronfluence/tree/main/examples/cifar.

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
        batch: Tuple[torch.Tensor, torch.Tensor],
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        """Compute the training loss for the given batch and model."""
        inputs, labels = batch
        logits = model(inputs)
        if not sample:
            return F.cross_entropy(logits, labels, reduction="sum")
        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
            sampled_labels = torch.multinomial(probs, num_samples=1).flatten()
        return F.cross_entropy(logits, sampled_labels, reduction="sum")

    def compute_measurement(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        model: nn.Module,
    ) -> torch.Tensor:
        """Compute the influence measurement for the given batch and model."""
        inputs, labels = batch
        logits = model(inputs)

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

    def get_influence_tracked_modules(self) -> Optional[List[str]]:
        """Return the list of modules for influence computation."""
        return self._tracked_modules


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


class CausalLMTask(Task):
    """Causal-language-modeling task for Kronfluence.

    Based on the ``wikitext`` example in kronfluence:
    https://github.com/pomonam/kronfluence/tree/main/examples/wikitext —
    shifted next-token cross-entropy with ``ignore_index=-100``.

    Parameters
    ----------
    tracked_modules : Optional[List[str]]
        Names of modules to restrict influence computation to (as in
        ``model.named_modules()``). If None, kronfluence tracks every
        applicable ``nn.Linear`` / ``nn.Conv2d``.

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
        ).logits.float()
        logits = logits[..., :-1, :].contiguous()
        logits = logits.view(-1, logits.size(-1))
        labels = batch["labels"][..., 1:].contiguous()
        if not sample:
            return F.cross_entropy(
                logits,
                labels.view(-1),
                reduction="sum",
                ignore_index=-100,
            )
        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
            sampled_labels = torch.multinomial(probs, num_samples=1).flatten()
            sampled_labels[labels.view(-1) == -100] = -100
        return F.cross_entropy(
            logits,
            sampled_labels,
            ignore_index=-100,
            reduction="sum",
        )

    def compute_measurement(
        self,
        batch: Dict[str, torch.Tensor],
        model: nn.Module,
    ) -> torch.Tensor:
        """Compute the influence measurement for the given batch and model."""
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits.float()
        shift_labels = batch["labels"][..., 1:].contiguous().view(-1)
        logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
        return F.cross_entropy(
            logits, shift_labels, ignore_index=-100, reduction="sum"
        )

    def get_attention_mask(
        self, batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Return the attention mask for the given batch."""
        return batch["attention_mask"]

    def get_influence_tracked_modules(self) -> Optional[List[str]]:
        """Return the list of modules for influence computation."""
        return self._tracked_modules


__all__ = [
    "ImageClassificationTask",
    "TextClassificationTask",
    "CausalLMTask",
]
