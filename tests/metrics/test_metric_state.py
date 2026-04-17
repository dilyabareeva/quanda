"""Tests for reset/state_dict/load_state_dict across all metrics."""

import pytest
import torch

from quanda.explainers.wrappers import CaptumSimilarity
from quanda.metrics.downstream_eval import (
    ClassDetectionMetric,
    MislabelingDetectionMetric,
    ShortcutDetectionMetric,
    SubclassDetectionMetric,
)
from quanda.metrics.ground_truth.linear_datamodeling import (
    LinearDatamodelingMetric,
)
from quanda.metrics.heuristics import (
    ModelRandomizationMetric,
    TopKCardinalityMetric,
)
from quanda.metrics.heuristics.mixed_datasets import MixedDatasetsMetric
from quanda.utils.functions import cosine_similarity
from quanda.utils.training import Trainer


def _class_detection(request, tmp_path):
    return ClassDetectionMetric(
        model=request.getfixturevalue("load_mnist_model"),
        checkpoints=request.getfixturevalue("load_mnist_last_checkpoint"),
        train_dataset=request.getfixturevalue("load_mnist_dataset"),
    )


def _subclass_detection(request, tmp_path):
    return SubclassDetectionMetric(
        model=request.getfixturevalue("load_mnist_model"),
        checkpoints=request.getfixturevalue("load_mnist_last_checkpoint"),
        train_dataset=request.getfixturevalue("load_grouped_mnist_dataset"),
        train_subclass_labels=request.getfixturevalue("load_mnist_labels"),
    )


def _shortcut_detection(request, tmp_path):
    return ShortcutDetectionMetric(
        model=request.getfixturevalue("load_mnist_model"),
        checkpoints=request.getfixturevalue("load_mnist_last_checkpoint"),
        train_dataset=request.getfixturevalue("load_mnist_dataset"),
        shortcut_indices=[3],
        shortcut_cls=1,
        filter_by_non_shortcut=False,
        filter_by_shortcut_pred=False,
    )


def _mislabeling_detection(request, tmp_path):
    dataset = request.getfixturevalue("load_mislabeling_mnist_dataset")
    return MislabelingDetectionMetric(
        model=request.getfixturevalue("load_mnist_model"),
        checkpoints=request.getfixturevalue("load_mnist_last_checkpoint"),
        train_dataset=dataset,
        mislabeling_indices=dataset.transform_indices,
        explainer_cls=CaptumSimilarity,
        expl_kwargs={
            "layers": "fc_2",
            "similarity_metric": cosine_similarity,
            "model_id": "test",
            "cache_dir": str(tmp_path),
        },
    )


def _mixed_datasets(request, tmp_path):
    return MixedDatasetsMetric(
        model=request.getfixturevalue("load_mnist_model"),
        checkpoints=request.getfixturevalue("load_mnist_last_checkpoint"),
        train_dataset=request.getfixturevalue("load_mnist_dataset"),
        adversarial_indices=request.getfixturevalue(
            "load_mnist_adversarial_indices"
        ),
    )


def _top_k_cardinality(request, tmp_path):
    return TopKCardinalityMetric(
        model=request.getfixturevalue("load_mnist_model"),
        checkpoints=request.getfixturevalue("load_mnist_last_checkpoint"),
        train_dataset=request.getfixturevalue("load_mnist_dataset"),
        top_k=3,
    )


def _model_randomization(request, tmp_path):
    return ModelRandomizationMetric(
        model=request.getfixturevalue("load_mnist_model"),
        model_id="0",
        checkpoints=request.getfixturevalue("load_mnist_last_checkpoint"),
        train_dataset=request.getfixturevalue("load_mnist_dataset"),
        explainer_cls=CaptumSimilarity,
        expl_kwargs={
            "layers": "fc_2",
            "similarity_metric": cosine_similarity,
            "model_id": "0",
            "cache_dir": str(tmp_path),
        },
        cache_dir=str(tmp_path),
        seed=42,
    )


def _linear_datamodeling(request, tmp_path):
    trainer = Trainer(
        max_epochs=1,
        optimizer=torch.optim.SGD,
        lr=0.1,
        criterion=torch.nn.CrossEntropyLoss(),
    )
    return LinearDatamodelingMetric(
        model=request.getfixturevalue("load_mnist_model"),
        checkpoints=request.getfixturevalue("load_mnist_last_checkpoint"),
        train_dataset=request.getfixturevalue("load_mnist_dataset"),
        trainer=trainer,
        alpha=0.5,
        model_id="mnist_lds",
        m=2,
        seed=3,
        correlation_fn="spearman",
        cache_dir=str(tmp_path),
        batch_size=1,
    )


@pytest.mark.heuristic_metrics
@pytest.mark.downstream_eval_metrics
@pytest.mark.ground_truth_metrics
@pytest.mark.parametrize(
    "factory",
    [
        _class_detection,
        _subclass_detection,
        _shortcut_detection,
        _mislabeling_detection,
        _mixed_datasets,
        _top_k_cardinality,
        _model_randomization,
        _linear_datamodeling,
    ],
)
def test_metric_state_roundtrip(factory, request, tmp_path):
    metric = factory(request, tmp_path)

    initial_state = metric.state_dict()
    metric.reset()
    reset_state = metric.state_dict()
    metric.load_state_dict(initial_state)
    restored_state = metric.state_dict()

    if initial_state is None:
        assert reset_state is None
        assert restored_state is None
    else:
        assert isinstance(initial_state, dict)
        assert isinstance(reset_state, dict)
        assert isinstance(restored_state, dict)
        assert set(restored_state.keys()) == set(initial_state.keys())
