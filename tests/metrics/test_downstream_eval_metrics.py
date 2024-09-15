import math

import pytest

from quanda.explainers import SumAggregator
from quanda.explainers.wrappers import CaptumSimilarity
from quanda.metrics.downstream_eval import (
    ClassDetectionMetric,
    DatasetCleaningMetric,
    MislabelingDetectionMetric,
    ShortcutDetectionMetric,
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
    # TODO: introduce a more meaningfull test, where the score is not zero
    # Note from Galip:
    # one idea could be: a random attributor should get approximately 1/( # of classes).
    # With a big test dataset, the probability of failing a truly random test
    # should diminish.
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
    "test_id, model, dataset, explanations, global_method, expl_kwargs, expected_score",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_poisoned_mnist_dataset",
            "load_mnist_explanations_similarity_1",
            "self-influence",
            {"layers": "fc_2", "similarity_metric": cosine_similarity, "model_id": "test", "cache_dir": "cache"},
            0.4921875,
        ),
        (
            "mnist",
            "load_mnist_model",
            "load_poisoned_mnist_dataset",
            "load_mnist_explanations_similarity_1",
            SumAggregator,
            None,
            0.4921875,
        ),
        (
            "mnist",
            "load_mnist_model",
            "load_poisoned_mnist_dataset",
            "load_mnist_explanations_similarity_1",
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
    global_method,
    expl_kwargs,
    expected_score,
    request,
):
    dataset = request.getfixturevalue(dataset)
    tda = request.getfixturevalue(explanations)
    model = request.getfixturevalue(model)
    if global_method != "self-influence":
        metric = MislabelingDetectionMetric(
            model=model,
            train_dataset=dataset,
            poisoned_indices=dataset.transform_indices,
            global_method=global_method,
            device="cpu",
        )
        metric.update(explanations=tda)
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


@pytest.mark.downstream_eval_metrics
@pytest.mark.parametrize(
    "test_id, model, dataset, labels, poisoned_ids, explanations",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_labels",
            [3],
            "load_mnist_explanations_similarity_1",
        ),
    ],
)
def test_shortcut_detection_metric(
    test_id,
    model,
    dataset,
    labels,
    poisoned_ids,
    explanations,
    request,
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    labels = request.getfixturevalue(labels)
    tda = request.getfixturevalue(explanations)
    poisoned_cls = labels[poisoned_ids[0]]
    metric = ShortcutDetectionMetric(model, dataset, poisoned_ids, poisoned_cls)
    metric.update(tda)
    res = metric.compute()
    poisoned = tda[:, poisoned_ids].mean()
    clean_ids = [i for i in range(len(labels)) if i not in poisoned_ids and labels[i] == poisoned_cls]
    clean = tda[:, clean_ids].mean()
    rest_indices = list(set(range(metric.dataset_length)) - set(poisoned_ids) - set(clean_ids))
    rest = tda[:, rest_indices].mean()
    assertions = [
        math.isclose(res["poisoned"], poisoned, abs_tol=0.00001),
        math.isclose(res["clean"], clean, abs_tol=0.00001),
        math.isclose(res["rest"], rest, abs_tol=0.00001),
    ]
    assert assertions[0], "Expected (poisoned): {}, Got: {}".format(poisoned, res["poisoned"])
    assert assertions[1], "Expected (clean): {}, Got: {}".format(clean, res["clean"])
    assert assertions[2], "Expected (rest): {}, Got: {}".format(rest, res["rest"])
    assert all(assertions)
