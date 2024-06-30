import pytest

from src.explainers.aggregators import SumAggregator
from src.explainers.wrappers.captum_influence import CaptumSimilarity
from src.metrics.localization.identical_class import IdenticalClass
from src.metrics.localization.identical_subclass import IdenticalSubclass
from src.metrics.localization.mislabeling_detection import (
    MislabelingDetectionMetric,
)
from src.utils.functions.similarities import cosine_similarity


@pytest.mark.localization_metrics
@pytest.mark.parametrize(
    "test_id,model,dataset,test_labels,batch_size,explanations,expected_score",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_labels_1",
            8,
            "load_mnist_explanations_1",
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
    metric = IdenticalClass(model=model, train_dataset=dataset, device="cpu")
    metric.update(test_labels=test_labels, explanations=tda)
    score = metric.compute()
    # TODO: introduce a more meaningfull test, where the score is not zero
    # Note from Galip:
    # one idea could be: a random attributor should get approximately 1/( # of classes).
    # With a big test dataset, the probability of failing a truly random test
    # should diminish.
    assert score == expected_score


@pytest.mark.localization_metrics
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
            "load_mnist_explanations_1",
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
    metric = IdenticalSubclass(
        model=model,
        train_dataset=dataset,
        subclass_labels=subclass_labels,
        device="cpu",
    )
    metric.update(test_subclasses=test_labels, explanations=tda)
    score = metric.compute()
    assert score == expected_score


@pytest.mark.localization_metrics
@pytest.mark.parametrize(
    "test_id, model, dataset, batch_size, explanations, global_method, explainer_kwargs, expected_score",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_poisoned_mnist_dataset",
            8,
            "load_mnist_explanations_1",
            "self-influence",
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.4375,  # = 28/64
        ),
        (
            "mnist",
            "load_mnist_model",
            "load_poisoned_mnist_dataset",
            8,
            "load_mnist_explanations_1",
            SumAggregator,
            None,
            0.4375,
        ),
        (
            "mnist",
            "load_mnist_model",
            "load_poisoned_mnist_dataset",
            8,
            "load_mnist_explanations_1",
            "sum_abs",
            None,
            0.4375,
        ),
    ],
)
def test_poisoning_detection_metric(
    test_id,
    model,
    dataset,
    batch_size,
    explanations,
    global_method,
    explainer_kwargs,
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
        explainer = CaptumSimilarity(
            model=model,
            model_id=test_id,
            cache_dir=str(0),
            train_dataset=dataset,
            device="cpu",
            **explainer_kwargs,
        )

        metric = MislabelingDetectionMetric(
            model=model,
            train_dataset=dataset,
            global_method=global_method,
            poisoned_indices=dataset.transform_indices,
            explainer=explainer,
            expl_kwargs={"batch_size": batch_size},
            device="cpu",
        )
    score = metric.compute()

    assert score["score"] == expected_score
