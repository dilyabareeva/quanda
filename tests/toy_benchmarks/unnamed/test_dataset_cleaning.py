import math

import lightning as L
import pytest

from quanda.explainers.wrappers.captum_influence import CaptumSimilarity
from quanda.toy_benchmarks.unnamed.dataset_cleaning import DatasetCleaning
from quanda.utils.functions.similarities import cosine_similarity
from quanda.utils.training.trainer import Trainer


@pytest.mark.toy_benchmarks
@pytest.mark.parametrize(
    "test_id, init_method, model, optimizer, lr, criterion, max_epochs, dataset, n_classes, n_groups, seed, "
    "global_method, batch_size, explainer_cls, expl_kwargs, use_pred, load_path, expected_score",
    [
        (
            "mnist1",
            "generate",
            "load_mnist_model",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            3,
            "load_mnist_dataset",
            10,
            2,
            27,
            "self-influence",
            8,
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            False,
            None,
            0.0,
        ),
        (
            "mnist2",
            "assemble",
            "load_mnist_model",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            3,
            "load_mnist_dataset",
            10,
            2,
            27,
            "sum",
            8,
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            False,
            None,
            0.0,
        ),
        (
            "mnist3",
            "load",
            "load_mnist_model",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            3,
            "load_mnist_dataset",
            10,
            2,
            27,
            "sum_abs",
            8,
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            False,
            "tests/assets/mnist_dataset_cleaning_state_dict",
            0.0,
        ),
    ],
)
def test_dataset_cleaning(
    test_id,
    init_method,
    model,
    optimizer,
    lr,
    criterion,
    max_epochs,
    dataset,
    n_classes,
    n_groups,
    seed,
    global_method,
    batch_size,
    explainer_cls,
    expl_kwargs,
    use_pred,
    load_path,
    expected_score,
    tmp_path,
    request,
):
    model = request.getfixturevalue(model)
    optimizer = request.getfixturevalue(optimizer)
    criterion = request.getfixturevalue(criterion)
    dataset = request.getfixturevalue(dataset)

    if init_method == "generate":
        dst_eval = DatasetCleaning.generate(
            model=model,
            train_dataset=dataset,
            device="cpu",
        )

    elif init_method == "assemble":
        dst_eval = DatasetCleaning.assemble(
            model=model,
            train_dataset=dataset,
        )
    else:
        raise ValueError(f"Invalid init_method: {init_method}")

    trainer = Trainer(
        max_epochs=max_epochs,
        optimizer=optimizer,
        lr=lr,
        criterion=criterion,
    )
    score = dst_eval.evaluate(
        expl_dataset=dataset,
        explainer_cls=explainer_cls,
        trainer=trainer,
        expl_kwargs=expl_kwargs,
        trainer_fit_kwargs={"max_epochs": max_epochs},
        cache_dir=str(tmp_path),
        model_id="default_model_id",
        use_predictions=use_pred,
        global_method=global_method,
        batch_size=batch_size,
        device="cpu",
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)


@pytest.mark.toy_benchmarks
@pytest.mark.parametrize(
    "test_id, pl_module, max_epochs, dataset, n_classes, n_groups, seed, "
    "global_method, batch_size, explainer_cls, expl_kwargs, use_pred, load_path, expected_score",
    [
        (
            "mnist1",
            "load_mnist_pl_module",
            3,
            "load_mnist_dataset",
            10,
            2,
            27,
            "self-influence",
            8,
            CaptumSimilarity,
            {
                "layers": "model.fc_2",
                "similarity_metric": cosine_similarity,
            },
            False,
            None,
            0.0,
        ),
    ],
)
def test_dataset_cleaning_generate_from_pl_module(
    test_id,
    pl_module,
    max_epochs,
    dataset,
    n_classes,
    n_groups,
    seed,
    global_method,
    batch_size,
    explainer_cls,
    expl_kwargs,
    use_pred,
    load_path,
    expected_score,
    tmp_path,
    request,
):
    pl_module = request.getfixturevalue(pl_module)
    dataset = request.getfixturevalue(dataset)

    trainer = L.Trainer(max_epochs=max_epochs)

    dst_eval = DatasetCleaning.generate(
        model=pl_module,
        train_dataset=dataset,
        device="cpu",
    )

    score = dst_eval.evaluate(
        expl_dataset=dataset,
        explainer_cls=explainer_cls,
        trainer=trainer,
        expl_kwargs=expl_kwargs,
        trainer_fit_kwargs={},
        cache_dir=str(tmp_path),
        model_id="default_model_id",
        use_predictions=use_pred,
        global_method=global_method,
        batch_size=batch_size,
        device="cpu",
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)
