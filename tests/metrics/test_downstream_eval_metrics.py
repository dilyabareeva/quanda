import math

import pytest

from quanda.explainers import SumAggregator
from quanda.explainers.wrappers import CaptumSimilarity
from quanda.metrics.downstream_eval import (
    ClassDetectionMetric,
    DatasetCleaningMetric,
    MislabelingDetectionMetric,
    SubclassDetectionMetric,
)
from quanda.utils.functions import cosine_similarity
from quanda.utils.training import Trainer


@pytest.mark.downstream_eval_metrics
@pytest.mark.parametrize(
    "test_id,model,dataset,test_labels,batch_size,explanations,expected_score",
    [
        (
            "mnist",
            "load_mnist_model",
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
    dataset,
    test_labels,
    batch_size,
    explanations,
    expected_score,
    request,
):
    model = request.getfixturevalue(model)
    test_labels = request.getfixturevalue(test_labels)
    dataset = request.getfixturevalue(dataset)
    tda = request.getfixturevalue(explanations)
    metric = ClassDetectionMetric(model=model, train_dataset=dataset, device="cpu")
    metric.update(test_labels=test_labels, explanations=tda)
    score = metric.compute()["score"]
    assert math.isclose(score, expected_score, abs_tol=0.00001)


@pytest.mark.downstream_eval_metrics
@pytest.mark.parametrize(
    "test_id, model, dataset, subclass_labels, test_labels, batch_size, explanations, expected_score",
    [
        (
            "mnist",
            "load_mnist_model",
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
    dataset,
    subclass_labels,
    test_labels,
    batch_size,
    explanations,
    expected_score,
    request,
):
    model = request.getfixturevalue(model)
    test_labels = request.getfixturevalue(test_labels)
    subclass_labels = request.getfixturevalue(subclass_labels)
    dataset = request.getfixturevalue(dataset)
    tda = request.getfixturevalue(explanations)
    metric = SubclassDetectionMetric(
        model=model,
        train_dataset=dataset,
        subclass_labels=subclass_labels,
        device="cpu",
    )
    metric.update(test_subclasses=test_labels, explanations=tda)
    score = metric.compute()["score"]
    assert math.isclose(score, expected_score, abs_tol=0.00001)


@pytest.mark.downstream_eval_metrics
@pytest.mark.parametrize(
    "test_id, model, dataset, explanations, test_samples, test_labels, global_method, expl_kwargs, expected_score",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_poisoned_mnist_dataset",
            "load_mnist_explanations_similarity_1",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            "self-influence",
            {"layers": "fc_2", "similarity_metric": cosine_similarity, "model_id": "test", "cache_dir": "cache"},
            0.4921875,
        ),
        (
            "mnist",
            "load_mnist_model",
            "load_poisoned_mnist_dataset",
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
            "load_poisoned_mnist_dataset",
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
    dataset,
    explanations,
    test_samples,
    test_labels,
    global_method,
    expl_kwargs,
    expected_score,
    request,
):
    dataset = request.getfixturevalue(dataset)
    tda = request.getfixturevalue(explanations)
    model = request.getfixturevalue(model)
    test_labels = request.getfixturevalue(test_labels)
    test_samples = request.getfixturevalue(test_samples)

    if global_method != "self-influence":
        metric = MislabelingDetectionMetric(
            model=model,
            train_dataset=dataset,
            poisoned_indices=dataset.transform_indices,
            global_method=global_method,
            device="cpu",
        )
        metric.update(test_data=test_samples, test_labels=test_labels, explanations=tda)
    else:
        metric = MislabelingDetectionMetric(
            model=model,
            train_dataset=dataset,
            global_method=global_method,
            poisoned_indices=dataset.transform_indices,
            explainer_cls=CaptumSimilarity,
            expl_kwargs=expl_kwargs,
            device="cpu",
        )
    score = metric.compute()["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)


@pytest.mark.downstream_eval_metrics
@pytest.mark.parametrize(
    "test_id,model,optimizer, lr, criterion, max_epochs,dataset,explanations,global_method,top_k,expl_kwargs,"
    "batch_size,expected_score",
    [
        (
            "mnist",
            "load_mnist_model",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            3,
            "load_mnist_dataset",
            "load_mnist_explanations_similarity_1",
            "sum_abs",
            50,
            None,
            None,
            0.0,
        ),
        (
            "mnist",
            "load_mnist_model",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            3,
            "load_mnist_dataset",
            "load_mnist_explanations_similarity_1",
            "self-influence",
            50,
            {"layers": "fc_2", "similarity_metric": cosine_similarity, "cache_dir": "cache", "model_id": "test"},
            8,
            0.0,
        ),
    ],
)
def test_dataset_cleaning(
    test_id,
    model,
    optimizer,
    lr,
    criterion,
    max_epochs,
    dataset,
    explanations,
    global_method,
    top_k,
    expl_kwargs,
    batch_size,
    expected_score,
    request,
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    explanations = request.getfixturevalue(explanations)
    optimizer = request.getfixturevalue(optimizer)
    criterion = request.getfixturevalue(criterion)

    trainer = Trainer(
        max_epochs=max_epochs,
        optimizer=optimizer,
        lr=lr,
        criterion=criterion,
    )

    if global_method != "self-influence":
        metric = DatasetCleaningMetric(
            model=model,
            train_dataset=dataset,
            global_method=global_method,
            trainer=trainer,
            trainer_fit_kwargs={"max_epochs": max_epochs},
            top_k=top_k,
            device="cpu",
        )

        metric.update(explanations=explanations)

    else:
        expl_kwargs = expl_kwargs or {}

        metric = DatasetCleaningMetric(
            model=model,
            train_dataset=dataset,
            global_method=global_method,
            trainer=trainer,
            trainer_fit_kwargs={"max_epochs": max_epochs},
            top_k=top_k,
            explainer_cls=CaptumSimilarity,
            expl_kwargs=expl_kwargs,
            device="cpu",
        )

    score = metric.compute()["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)


@pytest.mark.downstream_eval_metrics
@pytest.mark.parametrize(
    "test_id,model,optimizer, lr, criterion, max_epochs,dataset,explanations,top_k,expl_kwargs," "batch_size,expected_score",
    [
        (
            "mnist",
            "load_mnist_model",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            3,
            "load_mnist_dataset",
            "load_mnist_explanations_similarity_1",
            50,
            {"layers": "fc_2", "similarity_metric": cosine_similarity, "cache_dir": "cache", "model_id": "test"},
            8,
            0.0,
        ),
    ],
)
def test_dataset_cleaning_self_influence_based(
    test_id,
    model,
    optimizer,
    lr,
    criterion,
    max_epochs,
    dataset,
    explanations,
    top_k,
    expl_kwargs,
    batch_size,
    expected_score,
    request,
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    explanations = request.getfixturevalue(explanations)
    optimizer = request.getfixturevalue(optimizer)
    criterion = request.getfixturevalue(criterion)

    trainer = Trainer(
        max_epochs=max_epochs,
        optimizer=optimizer,
        lr=lr,
        criterion=criterion,
    )

    expl_kwargs = expl_kwargs or {}

    metric = DatasetCleaningMetric.self_influence_based(
        model=model,
        train_dataset=dataset,
        trainer=trainer,
        trainer_fit_kwargs={"max_epochs": max_epochs},
        top_k=top_k,
        explainer_cls=CaptumSimilarity,
        expl_kwargs=expl_kwargs,
        expplainer_kwargs={"batch_size": batch_size},
        device="cpu",
    )

    score = metric.compute()["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)


@pytest.mark.downstream_eval_metrics
@pytest.mark.parametrize(
    "test_id,model,optimizer, lr, criterion, max_epochs,dataset,explanations,top_k," "expected_score",
    [
        (
            "mnist",
            "load_mnist_model",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            3,
            "load_mnist_dataset",
            "load_mnist_explanations_similarity_1",
            50,
            0.0,
        ),
    ],
)
def test_dataset_cleaning_aggr_based(
    test_id,
    model,
    optimizer,
    lr,
    criterion,
    max_epochs,
    dataset,
    explanations,
    top_k,
    expected_score,
    request,
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    explanations = request.getfixturevalue(explanations)
    optimizer = request.getfixturevalue(optimizer)
    criterion = request.getfixturevalue(criterion)

    trainer = Trainer(
        max_epochs=max_epochs,
        optimizer=optimizer,
        lr=lr,
        criterion=criterion,
    )

    metric = DatasetCleaningMetric.aggr_based(
        model=model,
        train_dataset=dataset,
        trainer=trainer,
        aggregator_cls="sum_abs",
        trainer_fit_kwargs={"max_epochs": max_epochs},
        top_k=top_k,
        device="cpu",
    )

    metric.update(explanations=explanations)

    score = metric.compute()["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)
