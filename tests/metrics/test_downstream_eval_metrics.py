import math

import pytest
import torch

from quanda.explainers import SumAggregator
from quanda.explainers.wrappers import CaptumSimilarity
from quanda.metrics.downstream_eval import (
    ClassDetectionMetric,
    MislabelingDetectionMetric,
    MRRMetric,
    RecallAtKMetric,
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
    test_targets = request.getfixturevalue(test_labels)
    dataset = request.getfixturevalue(dataset)
    tda = request.getfixturevalue(explanations)
    metric = ClassDetectionMetric(
        model=model,
        checkpoints=checkpoint,
        train_dataset=dataset,
    )
    metric.update(test_targets=test_targets, explanations=tda)
    score = metric.compute()["score"]
    assert math.isclose(score, expected_score, abs_tol=0.00001)


@pytest.mark.downstream_eval_metrics
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, subclass_labels, test_labels, batch_size, explanations, filter_by_prediction, expected_score",
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
            False,
            0.1,
        ),
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_grouped_mnist_dataset",
            "load_mnist_labels",
            "load_mnist_test_labels_1",
            8,
            "load_mnist_explanations_similarity_1",
            True,
            ValueError,
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
    filter_by_prediction,
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
        filter_by_prediction=filter_by_prediction,
    )
    if isinstance(expected_score, type):
        with pytest.raises(expected_score):
            metric.update(test_labels=test_labels, explanations=tda)
        return
    metric.update(test_labels=test_labels, explanations=tda)
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


@pytest.mark.tested
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, labels, poisoned_ids, poisoned_cls, explanations, filter_by_prediction, expected",
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
            0.39000001549720764,
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
            False,
            AssertionError,
        ),
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_labels",
            [3],
            1,
            "load_mnist_explanations_similarity_1",
            True,
            ValueError,
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
    filter_by_prediction,
    expected,
    request,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    dataset = request.getfixturevalue(dataset)
    labels = request.getfixturevalue(labels)
    tda = request.getfixturevalue(explanations)
    if isinstance(expected, type):
        with pytest.raises(expected):
            ShortcutDetectionMetric(
                model,
                dataset,
                poisoned_ids,
                poisoned_cls,
                checkpoints=checkpoint,
                filter_by_prediction=filter_by_prediction,
            ).update(tda)
        return

    metric = ShortcutDetectionMetric(
        model,
        dataset,
        poisoned_ids,
        poisoned_cls,
        checkpoints=checkpoint,
        filter_by_prediction=filter_by_prediction,
    )
    metric.update(tda)
    score = metric.compute()["score"]
    assert math.isclose(score, expected, abs_tol=0.00001)


@pytest.mark.downstream_eval_metrics
@pytest.mark.parametrize(
    "test_id, model, checkpoint, dataset, explanations, expected_score",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_explanations_similarity_1",
            0.07678571343421936,
        ),
    ],
)
def test_mrr_metric(
    test_id,
    model,
    checkpoint,
    dataset,
    explanations,
    expected_score,
    request,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    dataset = request.getfixturevalue(dataset)
    tda = request.getfixturevalue(explanations)

    num_training_examples = tda.shape[1]
    num_queries = tda.shape[0]
    entailment_labels = torch.zeros(
        (num_queries, num_training_examples), dtype=torch.bool
    )
    entailment_labels[0, 1] = True
    entailment_labels[1, 0] = True
    entailment_labels[2, 2] = True

    metric = MRRMetric(
        model=model,
        checkpoints=checkpoint,
        train_dataset=dataset,
    )
    metric.update(explanations=tda, entailment_labels=entailment_labels)
    score = metric.compute()["score"]
    assert math.isclose(score, expected_score, abs_tol=0.00001)


@pytest.mark.downstream_eval_metrics
@pytest.mark.parametrize(
    "test_id, model, checkpoint, dataset, explanations, k, expected_score",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_explanations_similarity_1",
            2,
            0.10000000149011612,
        ),
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_explanations_similarity_1",
            4,
            0.20000000298023224,
        ),
    ],
)
def test_recall_at_k_metric(
    test_id,
    model,
    checkpoint,
    dataset,
    explanations,
    k,
    expected_score,
    request,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    dataset = request.getfixturevalue(dataset)
    tda = request.getfixturevalue(explanations)

    num_training_examples = tda.shape[1]
    num_queries = tda.shape[0]
    entailment_labels = torch.zeros(
        (num_queries, num_training_examples), dtype=torch.bool
    )
    entailment_labels[0, 1] = True
    entailment_labels[1, 0] = True
    entailment_labels[2, k + 1] = True

    metric = RecallAtKMetric(
        model=model, checkpoints=checkpoint, train_dataset=dataset, k=k
    )
    metric.update(explanations=tda, entailment_labels=entailment_labels)
    score = metric.compute()["score"]
    assert math.isclose(score, expected_score, abs_tol=0.00001)
