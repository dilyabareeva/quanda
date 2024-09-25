import math

import lightning as L
import pytest
import torch

from quanda.benchmarks.downstream_eval import ShortcutDetection
from quanda.explainers.wrappers.captum_influence import CaptumSimilarity
from quanda.utils.functions.similarities import cosine_similarity
from quanda.utils.training.trainer import Trainer


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, init_method, model, optimizer, lr, criterion, max_epochs, dataset, sample_fn, n_classes, shortcut_cls,"
    "shortcut_indices, p, seed, batch_size, explainer_cls, expl_kwargs, filter_by_class, expected_score",
    [
        (
            "mnist_generate",
            "generate",
            "load_mnist_model",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            0,
            "load_mnist_dataset",
            "mnist_white_square_transformation",
            10,
            1,
            None,
            1.0,
            27,
            8,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            True,
            0.0,
        ),
        (
            "mnist_assemble",
            "assemble",
            "load_mnist_model",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            0,
            "load_mnist_dataset",
            "mnist_white_square_transformation",
            10,
            1,
            [3, 6],
            1.0,
            27,
            8,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            False,
            1.0,
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
    shortcut_cls,
    shortcut_indices,
    p,
    seed,
    batch_size,
    explainer_cls,
    expl_kwargs,
    filter_by_class,
    expected_score,
    tmp_path,
    request,
):
    model = request.getfixturevalue(model)
    optimizer = request.getfixturevalue(optimizer)
    criterion = request.getfixturevalue(criterion)
    dataset = request.getfixturevalue(dataset)
    sample_fn = request.getfixturevalue(sample_fn)
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
            eval_dataset=dataset,
            filter_by_class=filter_by_class,
            filter_by_prediction=True,
            shortcut_cls=shortcut_cls,
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
            eval_dataset=dataset,
            filter_by_class=filter_by_class,
            filter_by_prediction=True,
            p=p,
            shortcut_cls=shortcut_cls,
            shortcut_indices=shortcut_indices,
            sample_fn=sample_fn,
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"Invalid init_method: {init_method}")

    results = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=batch_size,
    )
    assert math.isclose(results["score"], expected_score, abs_tol=0.00001)


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, pl_module, optimizer, lr, criterion, max_epochs, dataset, sample_fn, n_classes, shortcut_cls,"
    "shortcut_indices, p, seed, batch_size, explainer_cls, expl_kwargs, expected_score",
    [
        (
            "mnist",
            "load_mnist_pl_module",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            0,
            "load_mnist_dataset",
            "mnist_white_square_transformation",
            10,
            1,
            None,
            1.0,
            27,
            8,
            CaptumSimilarity,
            {"layers": "model.fc_2", "similarity_metric": cosine_similarity},
            0.29325,
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
    shortcut_cls,
    shortcut_indices,
    p,
    seed,
    batch_size,
    explainer_cls,
    expl_kwargs,
    expected_score,
    tmp_path,
    request,
):
    pl_module = request.getfixturevalue(pl_module)
    dataset = request.getfixturevalue(dataset)
    expl_kwargs = {**expl_kwargs, "model_id": "test", "cache_dir": str(tmp_path)}
    trainer = L.Trainer(max_epochs=max_epochs)
    sample_fn = request.getfixturevalue(sample_fn)
    dst_eval = ShortcutDetection.generate(
        model=pl_module,
        trainer=trainer,
        train_dataset=dataset,
        n_classes=n_classes,
        eval_dataset=dataset,
        filter_by_class=True,
        filter_by_prediction=False,
        shortcut_cls=shortcut_cls,
        p=p,
        sample_fn=sample_fn,
        trainer_fit_kwargs={},
        seed=seed,
        batch_size=batch_size,
    )

    results = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=batch_size,
    )
    assert math.isclose(results["score"], expected_score, abs_tol=0.00001)


@pytest.mark.tested
@pytest.mark.parametrize(
    "test_id, benchmark_name, batch_size, explainer_cls, expl_kwargs, expected_score",
    [
        (
            "mnist",
            "mnist_shortcut_detection",
            8,
            CaptumSimilarity,
            {
                "layers": "model.fc_2",
                "similarity_metric": cosine_similarity,
                "load_from_disk": True,
            },
            0.875,
        ),
    ],
)
def test_shortcut_detection_download(
    test_id,
    benchmark_name,
    batch_size,
    explainer_cls,
    expl_kwargs,
    expected_score,
    tmp_path,
):
    dst_eval = ShortcutDetection.download(
        name=benchmark_name,
        cache_dir=str(tmp_path),
        device="cpu",
    )

    expl_kwargs = {"model_id": "0", "cache_dir": str(tmp_path), **expl_kwargs}
    dst_eval.train_dataset = torch.utils.data.Subset(dst_eval.train_dataset, list(range(16)))
    dst_eval.shortcut_dataset = torch.utils.data.Subset(dst_eval.shortcut_dataset, list(range(16)))
    dst_eval.eval_dataset = torch.utils.data.Subset(dst_eval.eval_dataset, list(range(16)))
    dst_eval.shortcut_indices = [i for i in dst_eval.shortcut_indices if i < 16]

    score = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=batch_size,
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)
