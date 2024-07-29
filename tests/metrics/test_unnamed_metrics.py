import pytest

from src.explainers.wrappers.captum_influence import CaptumSimilarity
from src.metrics.unnamed.dataset_cleaning import DatasetCleaningMetric
from src.metrics.unnamed.top_k_overlap import TopKOverlapMetric
from src.utils.functions.similarities import cosine_similarity
from src.utils.training.base_pl_module import BasicLightningModule
from src.utils.training.trainer import Trainer


@pytest.mark.unnamed_metrics
@pytest.mark.parametrize(
    "test_id, model, dataset, top_k, batch_size, explanations, expected_score",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            3,
            8,
            "load_mnist_explanations_1",
            7,
        ),
    ],
)
def test_top_k_overlap_metrics(
    test_id,
    model,
    dataset,
    top_k,
    batch_size,
    explanations,
    expected_score,
    request,
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    explanations = request.getfixturevalue(explanations)
    metric = TopKOverlapMetric(model=model, train_dataset=dataset, top_k=top_k, device="cpu")
    metric.update(explanations=explanations)
    score = metric.compute()
    assert score == expected_score


@pytest.mark.unnamed_metrics
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
            "load_mnist_explanations_1",
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
            "load_mnist_explanations_1",
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

    pl_module = BasicLightningModule(
        model=model,
        optimizer=optimizer,
        lr=lr,
        criterion=criterion,
    )
    trainer = Trainer.from_lightning_module(model, pl_module)

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

    score = metric.compute()

    assert score == expected_score


@pytest.mark.unnamed_metrics
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
            "load_mnist_explanations_1",
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

    pl_module = BasicLightningModule(
        model=model,
        optimizer=optimizer,
        lr=lr,
        criterion=criterion,
    )
    trainer = Trainer.from_lightning_module(model, pl_module)

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

    score = metric.compute()

    assert score == expected_score


@pytest.mark.unnamed_metrics
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
            "load_mnist_explanations_1",
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

    pl_module = BasicLightningModule(
        model=model,
        optimizer=optimizer,
        lr=lr,
        criterion=criterion,
    )
    trainer = Trainer.from_lightning_module(model, pl_module)

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

    score = metric.compute()

    assert score == expected_score
