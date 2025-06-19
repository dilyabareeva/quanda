import pytest
import torch
from lightning.pytorch import Trainer as LightningTrainer

from quanda.explainers.wrappers import CaptumSimilarity
from quanda.metrics.ground_truth.linear_datamodeling import (
    LinearDatamodelingMetric,
)
from quanda.utils.functions import cosine_similarity
from quanda.utils.training import Trainer


@pytest.mark.tested
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, test_data, test_labels, optimizer, criterion, expected, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            "torch_sgd_optimizer",
            "torch_cross_entropy_loss_object",
            0.25999,
            {"layers": "relu_4", "similarity_metric": cosine_similarity},
        ),
    ],
)
def test_linear_datamodeling(
    test_id,
    model,
    checkpoint,
    dataset,
    test_data,
    test_labels,
    optimizer,
    criterion,
    expected,
    method_kwargs,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    dataset = request.getfixturevalue(dataset)
    test_data = request.getfixturevalue(test_data)
    test_labels = request.getfixturevalue(test_labels)
    optimizer_cls = request.getfixturevalue(optimizer)
    criterion = request.getfixturevalue(criterion)

    trainer = Trainer(
        max_epochs=3,
        optimizer=optimizer_cls,
        lr=0.1,
        criterion=criterion,
    )

    explainer = CaptumSimilarity(
        model=model,
        checkpoints=checkpoint,
        model_id="mnist_similarity",
        cache_dir=str(tmp_path),
        train_dataset=dataset,
        device="cpu",
        **method_kwargs,
    )
    explanations = explainer.explain(test_data)
    explanation_targets = torch.tensor(test_labels)

    metric = LinearDatamodelingMetric(
        model=model,
        checkpoints=checkpoint,
        train_dataset=dataset,
        trainer=trainer,
        alpha=0.5,
        model_id="mnist_lds",
        m=5,
        seed=3,
        correlation_fn="spearman",
        cache_dir=str(tmp_path),
        batch_size=1,
    )

    metric.update(
        test_data=test_data,
        explanations=explanations,
        test_targets=explanation_targets,
    )

    score = metric.compute()["score"]

    assert abs(score - expected) < 0.01, "LDS scores differ significantly."


@pytest.mark.tested
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, test_data, test_labels, optimizer, criterion, "
    "trainer, correlation, expected, method_kwargs",
    [
        (
            "mnist_wrong_correlation",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            "torch_sgd_optimizer",
            "torch_cross_entropy_loss_object",
            "dummy_trainer",
            "Wrong Type Correlation",
            ValueError,
            {"layers": "relu_4", "similarity_metric": cosine_similarity},
        ),
        (
            "mnist_wrong_trainer",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            "torch_sgd_optimizer",
            "torch_cross_entropy_loss_object",
            None,
            "spearman",
            ValueError,
            {"layers": "relu_4", "similarity_metric": cosine_similarity},
        ),
    ],
)
def test_linear_datamodeling_edge_cases(
    test_id,
    model,
    checkpoint,
    dataset,
    test_data,
    test_labels,
    optimizer,
    criterion,
    trainer,
    correlation,
    expected,
    method_kwargs,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model) if model else None
    checkpoint = request.getfixturevalue(checkpoint)
    dataset = request.getfixturevalue(dataset)
    trainer = request.getfixturevalue(trainer) if trainer else None

    with pytest.raises(expected):
        LinearDatamodelingMetric(
            model=model,
            checkpoints=checkpoint,
            train_dataset=dataset,
            trainer=trainer,
            alpha=0.5,
            model_id="mnist_lds",
            m=5,
            seed=3,
            correlation_fn=correlation,
            cache_dir=str(tmp_path),
            batch_size=1,
        )

    return


@pytest.mark.tested
@pytest.mark.parametrize(
    "test_id, model, dataset, test_data, test_labels, subset_indices, pretrained_models, optimizer, criterion, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            "load_subset_indices_lds",
            "load_pretrained_models_lds",
            "torch_sgd_optimizer",
            "torch_cross_entropy_loss_object",
            {"layers": "relu_4", "similarity_metric": cosine_similarity},
        ),
    ],
)
def test_linear_datamodeling_extended(
    test_id,
    model,
    dataset,
    test_data,
    test_labels,
    optimizer,
    criterion,
    method_kwargs,
    request,
    tmp_path,
    subset_indices,
    pretrained_models,
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    test_data = request.getfixturevalue(test_data)
    test_labels = request.getfixturevalue(test_labels)
    optimizer_cls = request.getfixturevalue(optimizer)
    criterion = request.getfixturevalue(criterion)
    subset_indices = request.getfixturevalue(subset_indices)
    pretrained_models = request.getfixturevalue(pretrained_models)

    trainer = Trainer(
        max_epochs=3,
        optimizer=optimizer_cls,
        lr=0.1,
        criterion=criterion,
    )

    explainer = CaptumSimilarity(
        model=model,
        model_id="mnist_similarity",
        cache_dir=str(tmp_path),
        train_dataset=dataset,
        device="cpu",
        **method_kwargs,
    )
    explanations = explainer.explain(test_data)
    explanation_targets = torch.tensor(test_labels)

    metric = LinearDatamodelingMetric(
        model=model,
        train_dataset=dataset,
        trainer=trainer,
        alpha=0.5,
        model_id="mnist_lds",
        m=len(pretrained_models),
        seed=3,
        correlation_fn="spearman",
        cache_dir="tests/assets/lds_checkpoints/",
        batch_size=1,
        subset_ids=subset_indices,
        subset_ckpt_filenames=pretrained_models,
    )

    metric.update(
        test_data=test_data,
        explanations=explanations,
        test_targets=explanation_targets,
    )
    score = metric.compute()["score"]

    assert isinstance(score, float), "Score should be a float."


@pytest.mark.ground_truth_metrics
@pytest.mark.parametrize(
    "test_id, model, checkpoint, dataset, trainer",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "lightning_trainer",
        ),
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            None,
        ),
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "base_trainer",
        ),
    ],
)
def test_linear_datamodeling_training(
    test_id,
    model,
    checkpoint,
    dataset,
    trainer,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    dataset = request.getfixturevalue(dataset)

    if trainer == "lightning_trainer":
        lightning_trainer = LightningTrainer(
            max_epochs=0,
        )

        with pytest.raises(
            ValueError,
            match="Model should be a LightningModule if Trainer is a Lightning Trainer",
        ):
            LinearDatamodelingMetric(
                model=model,
                checkpoints=checkpoint,
                train_dataset=dataset,
                trainer=lightning_trainer,
                alpha=0.5,
                model_id=f"{test_id}_lds",
                m=5,
                seed=3,
                correlation_fn="spearman",
                cache_dir=str(tmp_path),
                batch_size=1,
            )

    elif trainer is None:
        lightning_trainer = None

        with pytest.raises(
            ValueError, match="Invalid combination of argumetns"
        ):
            LinearDatamodelingMetric(
                model=model,
                checkpoints=checkpoint,
                train_dataset=dataset,
                trainer=lightning_trainer,
                alpha=0.5,
                model_id=f"{test_id}_lds",
                m=5,
                seed=3,
            )

    elif trainer == "base_trainer":
        base_trainer = Trainer(
            max_epochs=3,
            optimizer=torch.optim.SGD,
            lr=0.1,
            criterion=torch.nn.CrossEntropyLoss(),
        )
        metric = LinearDatamodelingMetric(
            model=model,
            checkpoints=checkpoint,
            train_dataset=dataset,
            trainer=base_trainer,
            alpha=0.5,
            model_id=f"{test_id}_lds",
            m=5,
            seed=3,
            correlation_fn="spearman",
            cache_dir=str(tmp_path),
            batch_size=1,
        )

        assert isinstance(metric, LinearDatamodelingMetric), (
            "Metric should be an instance of LinearDatamodelingMetric."
        )
