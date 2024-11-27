import math

import pytest
import torch
from torcheval.metrics.functional import binary_auprc

from quanda.explainers import SumAggregator
from quanda.explainers.wrappers import CaptumSimilarity
from quanda.metrics.downstream_eval import (
    ClassDetectionMetric,
    MislabelingDetectionMetric,
    ShortcutDetectionMetric,
    SubclassDetectionMetric,
)
from quanda.utils.functions import cosine_similarity


@pytest.mark.downstream_eval_metrics
@pytest.mark.parametrize(
    "test_id,model,checkpoint,dataset,test_labels,batch_size,explanations,expected_score",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_test_labels_1",
            8,
            "load_mnist_explanations_similarity_1",
            0.1,
        ),
    ],
)
def test_identical_class_metrics(
    test_id,
    model,
    checkpoint,
    dataset,
    test_labels,
    batch_size,
    explanations,
    expected_score,
    request,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    test_labels = request.getfixturevalue(test_labels)
    dataset = request.getfixturevalue(dataset)
    tda = request.getfixturevalue(explanations)
    metric = ClassDetectionMetric(
        model=model,
        checkpoints=checkpoint,
        train_dataset=dataset,
    )
    metric.update(test_labels=test_labels, explanations=tda)
    score = metric.compute()["score"]
    assert math.isclose(score, expected_score, abs_tol=0.00001)


@pytest.mark.downstream_eval_metrics
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, subclass_labels, test_labels, batch_size, explanations, expected_score",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_grouped_mnist_dataset",
            "load_mnist_labels",
            "load_mnist_test_labels_1",
            8,
            "load_mnist_explanations_similarity_1",
            0.1,
        ),
    ],
)
def test_identical_subclass_metrics(
    test_id,
    model,
    checkpoint,
    dataset,
    subclass_labels,
    test_labels,
    batch_size,
    explanations,
    expected_score,
    request,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    test_labels = request.getfixturevalue(test_labels)
    subclass_labels = request.getfixturevalue(subclass_labels)
    dataset = request.getfixturevalue(dataset)
    tda = request.getfixturevalue(explanations)
    metric = SubclassDetectionMetric(
        model=model,
        checkpoints=checkpoint,
        train_dataset=dataset,
        train_subclass_labels=subclass_labels,
    )
    metric.update(test_subclasses=test_labels, explanations=tda)
    score = metric.compute()["score"]
    assert math.isclose(score, expected_score, abs_tol=0.00001)


@pytest.mark.downstream_eval_metrics
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, explanations, test_samples, test_labels, global_method, expl_kwargs, expected_score",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mislabeling_mnist_dataset",
            "load_mnist_explanations_similarity_1",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            "self-influence",
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
                "model_id": "test",
            },
            0.4921875,
        ),
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mislabeling_mnist_dataset",
            "load_mnist_explanations_similarity_1",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            SumAggregator,
            None,
            0.4921875,
        ),
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mislabeling_mnist_dataset",
            "load_mnist_explanations_similarity_1",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            "sum_abs",
            None,
            0.4921875,
        ),
    ],
)
def test_mislabeling_detection_metric(
    test_id,
    model,
    checkpoint,
    dataset,
    explanations,
    test_samples,
    test_labels,
    global_method,
    expl_kwargs,
    expected_score,
    request,
    tmp_path,
):
    dataset = request.getfixturevalue(dataset)
    tda = request.getfixturevalue(explanations)
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    test_labels = request.getfixturevalue(test_labels)
    test_samples = request.getfixturevalue(test_samples)

    if global_method != "self-influence":
        metric = MislabelingDetectionMetric(
            model=model,
            checkpoints=checkpoint,
            train_dataset=dataset,
            mislabeling_indices=dataset.transform_indices,
            global_method=global_method,
        )
        metric.update(
            test_data=test_samples, test_labels=test_labels, explanations=tda
        )
    else:
        metric = MislabelingDetectionMetric(
            model=model,
            checkpoints=checkpoint,
            train_dataset=dataset,
            global_method=global_method,
            mislabeling_indices=dataset.transform_indices,
            explainer_cls=CaptumSimilarity,
            expl_kwargs={**expl_kwargs, "cache_dir": str(tmp_path)},
        )
    score = metric.compute()["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)


@pytest.mark.downstream_eval_metrics
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, labels, poisoned_ids, poisoned_cls, explanations, assert_err",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_labels",
            [3],
            1,
            "load_mnist_explanations_similarity_1",
            False,
        ),
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_labels",
            [3],
            0,
            "load_mnist_explanations_similarity_1",
            True,
        ),
    ],
)
def test_shortcut_detection_metric(
    test_id,
    model,
    checkpoint,
    dataset,
    labels,
    poisoned_ids,
    poisoned_cls,
    explanations,
    assert_err,
    request,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    dataset = request.getfixturevalue(dataset)
    labels = request.getfixturevalue(labels)
    tda = request.getfixturevalue(explanations)
    if assert_err:
        with pytest.raises(AssertionError):
            metric = ShortcutDetectionMetric(
                model,
                dataset,
                poisoned_ids,
                poisoned_cls,
                checkpoints=checkpoint,
            )
    else:
        metric = ShortcutDetectionMetric(
            model,
            dataset,
            poisoned_ids,
            poisoned_cls,
            checkpoints=checkpoint,
        )
        metric.update(tda)
        score = metric.compute()["score"]
        binary_ids = torch.tensor(
            [1 if i in poisoned_ids else 0 for i in range(len(dataset))]
        )
        expected_score = (
            torch.tensor(
                [binary_auprc(tda[i], binary_ids) for i in range(tda.shape[0])]
            )
            .mean()
            .item()
        )
        assert math.isclose(score, expected_score, abs_tol=0.00001)
