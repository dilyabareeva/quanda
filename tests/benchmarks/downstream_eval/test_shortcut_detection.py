import math

import lightning as L
import pytest

from quanda.benchmarks.downstream_eval import ShortcutDetection
from quanda.explainers.aggregators import SumAggregator
from quanda.explainers.wrappers.captum_influence import CaptumSimilarity
from quanda.utils.functions.similarities import cosine_similarity
from quanda.utils.training.trainer import Trainer


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, init_method, model, optimizer, lr, criterion, max_epochs, dataset, sample_fn, n_classes, poisoned_cls,"
    "poisoned_indices, p, seed, batch_size, explainer_cls, expl_kwargs, expected_scores",
    [
        (
            "mnist",
            "generate",
            "load_mnist_model",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            0,
            "load_mnist_dataset",
            "box_1c",
            10,
            1,
            None,
            1.0,
            27,
            8,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            {"score": 0.32555, "clean": 0.0, "rest": 0.29015},
        ),
        (
            "mnist",
            "assemble",
            "load_mnist_model",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            0,
            "load_mnist_dataset",
            "box_1c",
            10,
            1,
            [3, 6],
            1.0,
            27,
            8,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            {"score": 0.32555, "clean": 0.0, "rest": 0.29015},
        ),
    ],
)
def test_shortcut_detection(
    test_id,
    init_method,
    model,
    optimizer,
    lr,
    criterion,
    max_epochs,
    dataset,
    sample_fn,
    n_classes,
    poisoned_cls,
    poisoned_indices,
    p,
    seed,
    batch_size,
    explainer_cls,
    expl_kwargs,
    expected_scores,
    tmp_path,
    request,
):
    model = request.getfixturevalue(model)
    optimizer = request.getfixturevalue(optimizer)
    criterion = request.getfixturevalue(criterion)
    dataset = request.getfixturevalue(dataset)
    expl_kwargs = {**expl_kwargs, "model_id": "test", "cache_dir": str(tmp_path)}
    if init_method == "generate":
        trainer = Trainer(
            max_epochs=max_epochs,
            optimizer=optimizer,
            lr=lr,
            criterion=criterion,
        )

        dst_eval = ShortcutDetection.generate(
            model=model,
            trainer=trainer,
            train_dataset=dataset,
            n_classes=n_classes,
            poisoned_cls=poisoned_cls,
            p=p,
            sample_fn=sample_fn,
            trainer_fit_kwargs={"max_epochs": max_epochs},
            seed=seed,
            batch_size=batch_size,
        )

    elif init_method == "assemble":
        dst_eval = ShortcutDetection.assemble(
            model=model,
            train_dataset=dataset,
            n_classes=n_classes,
            p=p,
            poisoned_cls=poisoned_cls,
            poisoned_indices=poisoned_indices,
            sample_fn=sample_fn,
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"Invalid init_method: {init_method}")

    results = dst_eval.evaluate(
        expl_dataset=dataset,
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        cache_dir=str(tmp_path),
        model_id="default_model_id",
        batch_size=batch_size,
    )
    print(list(results.keys()))
    assertions = [math.isclose(results[k], expected_scores[k], abs_tol=0.00001) for k in expected_scores.keys()]
    assert all(assertions)


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, pl_module, optimizer, lr, criterion, max_epochs, dataset, sample_fn, n_classes, poisoned_cls,"
    "poisoned_indices, p, seed, batch_size, explainer_cls, expl_kwargs, expected_scores",
    [
        (
            "mnist",
            "load_mnist_pl_module",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            0,
            "load_mnist_dataset",
            "box_1c",
            10,
            1,
            None,
            1.0,
            27,
            8,
            CaptumSimilarity,
            {"layers": "model.fc_2", "similarity_metric": cosine_similarity},
            {"score": 0.32555, "clean": 0.0, "rest": 0.29015},
        ),
    ],
)
def test_shortcut_detection_generate_from_pl_module(
    test_id,
    pl_module,
    optimizer,
    lr,
    criterion,
    max_epochs,
    dataset,
    sample_fn,
    n_classes,
    poisoned_cls,
    poisoned_indices,
    p,
    seed,
    batch_size,
    explainer_cls,
    expl_kwargs,
    expected_scores,
    tmp_path,
    request,
):
    pl_module = request.getfixturevalue(pl_module)
    dataset = request.getfixturevalue(dataset)
    expl_kwargs = {**expl_kwargs, "model_id": "test", "cache_dir": str(tmp_path)}
    trainer = L.Trainer(max_epochs=max_epochs)

    dst_eval = ShortcutDetection.generate(
        model=pl_module,
        trainer=trainer,
        train_dataset=dataset,
        n_classes=n_classes,
        poisoned_cls=poisoned_cls,
        p=p,
        sample_fn=sample_fn,
        trainer_fit_kwargs={},
        seed=seed,
        batch_size=batch_size,
    )

    results = dst_eval.evaluate(
        expl_dataset=dataset,
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        cache_dir=str(tmp_path),
        model_id="default_model_id",
        batch_size=batch_size,
    )
    assertions = [math.isclose(results[k], expected_scores[k], abs_tol=0.00001) for k in expected_scores.keys()]
    assert all(assertions)
