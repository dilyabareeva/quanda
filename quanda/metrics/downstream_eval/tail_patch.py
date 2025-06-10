"""Tail Patch Metric."""

import copy
from typing import Any, Callable, List, Optional, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW, Optimizer
from transformers import AutoTokenizer  # type: ignore

from quanda.metrics.base import Metric


class TailPatchMetric(Metric):
    """Tail Patch Metric.

    This metric measures the incremental training probability increase by
    taking a single training step on retrieved proponents and measuring
    the change in target sequence probability.

    References
    ----------
    1) Tyler A. Chang, Dheeraj Rajagopal, Tolga Bolukbasi, Lucas Dixon,
    and Ian Tenney. (2024) "Scalable Influence and Fact Tracing for
    Large Language Model Pretraining". The Thirteenth International
    Conference on Learning Representations.

    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: torch.utils.data.Dataset,
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        k: int = 10,
        learning_rate: float = 1e-3,
        optimizer_class: Type[Optimizer] = AdamW,
        optimizer_kwargs: Optional[dict] = None,
        tokenizer_name: str = "gpt2",
    ):
        """Initialize the Tail Patch Metric.

        Parameters
        ----------
        model : nn.Module
            The model associated with the attributions to be evaluated.
        train_dataset : torch.utils.data.Dataset
            The training dataset that was used to train `model`.
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s), defaults to None.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file, takes
            (model, checkpoint path) as two arguments, by default None.
        k : int, optional
            The number of top proponents to evaluate, by default 10.
        learning_rate : float, optional
            Learning rate for the single gradient step, by default 1e-3.
        optimizer_class : Type[Optimizer], optional
            Optimizer class to use for updates, by default AdamW.
        optimizer_kwargs : Optional[dict], optional
            Additional keyword arguments for the optimizer, by default None.
        tokenizer_name : str, optional
            Name of the tokenizer to use for text decoding, by default "gpt2".

        """
        super().__init__(
            model=model,
            checkpoints=checkpoints,
            train_dataset=train_dataset,
            checkpoints_load_func=checkpoints_load_func,
        )
        self.k = k
        self.learning_rate = learning_rate
        self.device = next(model.parameters()).device
        self.scores: List[float] = []
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _clone_model(self) -> nn.Module:
        """Clone the model via deep copy and move to the correct device."""
        clone = copy.deepcopy(self.model)
        clone.to(self.device)
        return clone

    def _compute_log_prob(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the log-probability of the target tokens.

        Return sequence log-probabilities: sum log p(y_t|…)
        over target tokens only.
        """
        model.eval()
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        target_ids = target_ids.to(self.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
            )
            logits = outputs.logits  # [B, T, V]
            log_probs = F.log_softmax(logits, dim=-1)
            # clamp invalid indices before gather
            target_clamped = target_ids.clone()
            target_clamped[target_clamped < 0] = 0
            token_logps = log_probs.gather(
                2, target_clamped.unsqueeze(-1)
            ).squeeze(
                -1
            )  # [B, T]
            # Mask out non-target or padding tokens
            mask = (target_ids != -100) & attention_mask.bool()
            token_logps = token_logps * mask
            seq_logp = token_logps.sum(dim=1)  # [B]
        return seq_logp.double()

    def _single_step_update(
        self,
        model: nn.Module,
        proponent: dict,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Perform exactly one training step on `proponent` example."""
        model.train()
        optimizer.zero_grad()

        batch = {}
        for key, val in proponent.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val.to(self.device)
            elif isinstance(val, (list, np.ndarray)):
                batch[key] = torch.tensor(val, device=self.device)
            else:
                batch[key] = val

        allowed_keys = {"input_ids", "attention_mask", "labels"}
        clean_batch = {k: v for k, v in batch.items() if k in allowed_keys}
        outputs = model(**clean_batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    def update(
        self,
        explanations: torch.Tensor,
        test_data: dict,
        test_targets: torch.Tensor,
    ) -> None:
        """Update the metric state with the provided explanations.

        Parameters
        ----------
        explanations : torch.Tensor
            Attributions for training examples for each query.
        test_data : dict
            Dictionary containing test data with keys like 'input_ids',
            'attention_mask'.
        test_targets : torch.Tensor
            Target token IDs for computing probabilities, with -100 for
            non-target tokens.

        """
        input_ids = test_data["input_ids"].to(self.device)
        attention_mask = test_data["attention_mask"].to(self.device)
        test_targets = test_targets.to(self.device)

        # Original log-prob
        orig_logps = self._compute_log_prob(
            self.model,
            input_ids,
            attention_mask,
            test_targets,
        )
        orig_probs = torch.exp(orig_logps)

        # Top-k proponents
        k = min(self.k, explanations.size(1))
        topk_idx = torch.topk(explanations, k, dim=1).indices

        batch_size = input_ids.size(0)
        for i in range(batch_size):
            p_orig = orig_probs[i].item()
            deltas = []

            for idx in topk_idx[i].tolist():
                model_copy = self._clone_model()
                optimizer = self.optimizer_class(
                    [p for p in model_copy.parameters() if p.requires_grad],
                    lr=self.learning_rate,
                    **self.optimizer_kwargs,
                )

                proponent_data = self.train_dataset[idx]
                self._single_step_update(model_copy, proponent_data, optimizer)

                new_logp = self._compute_log_prob(
                    model_copy,
                    input_ids[i : i + 1],
                    attention_mask[i : i + 1],
                    test_targets[i : i + 1],
                )
                p_new = torch.exp(new_logp).item()
                delta = p_new - p_orig
                deltas.append(delta)

            self.scores.append(torch.tensor(deltas).mean().item())

    def compute(self) -> dict:
        """Compute the TailPatch metric.

        Returns
        -------
        Dict[str, float]
            A dictionary containing the metric score.

        """
        return {
            "score": (
                torch.tensor(self.scores).mean().item() if self.scores else 0.0
            )
        }

    def reset(self) -> None:
        """Reset the metric state."""
        self.scores = []

    def load_state_dict(self, state_dict: dict) -> None:
        """Load previously computed state for the metric.

        Parameters
        ----------
        state_dict : dict
            A state dictionary for the metric.

        """
        self.scores = state_dict.get("scores", [])

    def state_dict(self) -> dict:
        """Return the metric state.

        Returns
        -------
        dict
            The state dictionary containing the scores.

        """
        return {"scores": self.scores}
