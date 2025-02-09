import pytest
import torch
from lightning.pytorch import Trainer as LightningTrainer


from quanda.explainers.wrappers import CaptumSimilarity
from quanda.metrics.ground_truth.linear_datamodeling import (
    LinearDatamodelingMetric,
)
from quanda.utils.functions import cosine_similarity
from quanda.utils.training import Trainer


@pytest.mark.ground_truth_metrics
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, test_data, test_labels, optimizer, criterion, method_kwargs",
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
    get_lds_score,
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

    assert (
        abs(score - get_lds_score) < 0.01
    ), "LDS scores differ significantly."


@pytest.mark.ground_truth_metrics
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
        m=len(subset_indices),
        seed=3,
        correlation_fn="spearman",
        cache_dir=str(tmp_path),
        batch_size=1,
        subset_ids=subset_indices,
        pretrained_models=pretrained_models,
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
    "test_id, model, checkpoint, dataset",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
        ),
    ],
)
def test_linear_datamodeling_lightning(
    test_id,
    model,
    checkpoint,
    dataset,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    dataset = request.getfixturevalue(dataset)

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
