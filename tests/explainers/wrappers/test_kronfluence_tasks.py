"""Tests for ``quanda.explainers.wrappers.kronfluence_tasks``."""

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from quanda.explainers.wrappers.kronfluence_tasks import (
    ImageClassificationTask,
    TextClassificationTask,
)


class _LinearImageModel(nn.Module):
    def __init__(self, in_features: int = 4, num_classes: int = 3) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class _LinearTextModel(nn.Module):
    def __init__(self, vocab_size: int = 5, num_classes: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        embeds = self.embed(input_ids)
        mask = attention_mask.unsqueeze(-1).float()
        return (embeds * mask).sum(dim=1)


@pytest.mark.explainers
class TestImageClassificationTask:
    def _batch(self):
        torch.manual_seed(0)
        inputs = torch.randn(4, 4)
        labels = torch.tensor([0, 1, 2, 1])
        return inputs, labels

    def test_compute_train_loss_matches_cross_entropy(self):
        task = ImageClassificationTask()
        model = _LinearImageModel()
        inputs, labels = self._batch()

        loss = task.compute_train_loss((inputs, labels), model, sample=False)

        expected = F.cross_entropy(model(inputs), labels, reduction="sum")
        assert torch.allclose(loss, expected)

    def test_compute_train_loss_sample_branch_returns_scalar(self):
        task = ImageClassificationTask()
        model = _LinearImageModel()
        inputs, labels = self._batch()

        loss = task.compute_train_loss((inputs, labels), model, sample=True)

        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_compute_measurement_matches_margin_formula(self):
        task = ImageClassificationTask()
        model = _LinearImageModel()
        inputs, labels = self._batch()

        measurement = task.compute_measurement((inputs, labels), model)

        logits = model(inputs)
        bindex = torch.arange(logits.shape[0])
        correct = logits[bindex, labels]
        cloned = logits.clone()
        cloned[bindex, labels] = -torch.inf
        expected = -(correct - cloned.logsumexp(dim=-1)).sum()
        assert torch.allclose(measurement, expected)

    def test_get_influence_tracked_modules_default_is_none(self):
        assert (
            ImageClassificationTask().get_influence_tracked_modules() is None
        )

    def test_get_influence_tracked_modules_returns_provided_list(self):
        tracked = ["fc", "conv"]
        task = ImageClassificationTask(tracked_modules=tracked)
        assert task.get_influence_tracked_modules() == tracked


@pytest.mark.explainers
class TestTextClassificationTask:
    def _batch(self):
        torch.manual_seed(0)
        input_ids = torch.tensor([[1, 2, 3], [0, 1, 4]])
        attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        token_type_ids = torch.zeros_like(input_ids)
        labels = torch.tensor([0, 1])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }

    def test_compute_train_loss_matches_cross_entropy(self):
        task = TextClassificationTask()
        model = _LinearTextModel()
        batch = self._batch()

        loss = task.compute_train_loss(batch, model, sample=False)

        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        )
        expected = F.cross_entropy(logits, batch["labels"], reduction="sum")
        assert torch.allclose(loss, expected)

    def test_compute_train_loss_sample_branch_returns_scalar(self):
        task = TextClassificationTask()
        model = _LinearTextModel()
        batch = self._batch()

        loss = task.compute_train_loss(batch, model, sample=True)

        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_compute_measurement_matches_margin_formula(self):
        task = TextClassificationTask()
        model = _LinearTextModel()
        batch = self._batch()

        measurement = task.compute_measurement(batch, model)

        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        )
        bindex = torch.arange(logits.shape[0])
        correct = logits[bindex, batch["labels"]]
        cloned = logits.clone()
        cloned[bindex, batch["labels"]] = -torch.inf
        expected = -(correct - cloned.logsumexp(dim=-1)).sum()
        assert torch.allclose(measurement, expected)

    def test_get_attention_mask_returns_batch_mask(self):
        task = TextClassificationTask()
        batch = self._batch()

        mask = task.get_attention_mask(batch)

        assert torch.equal(mask, batch["attention_mask"])

    def test_get_influence_tracked_modules_default_is_none(self):
        assert TextClassificationTask().get_influence_tracked_modules() is None

    def test_get_influence_tracked_modules_returns_provided_list(self):
        tracked = ["embed"]
        task = TextClassificationTask(tracked_modules=tracked)
        assert task.get_influence_tracked_modules() == tracked
