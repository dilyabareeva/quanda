import math

import lightning as L
import pytest

from quanda.explainers.aggregators import SumAggregator
from quanda.explainers.wrappers.captum_influence import CaptumSimilarity
from quanda.toy_benchmarks.localization.mislabeling_detection import (
    MislabelingDetection,
)
from quanda.utils.functions.similarities import cosine_similarity
from quanda.utils.training.trainer import Trainer


@pytest.mark.toy_benchmarks
@pytest.mark.parametrize(
    "test_id, init_method, model, optimizer, lr, criterion, max_epochs, dataset, n_classes, p, seed, "
    "global_method, batch_size, explainer_cls, expl_kwargs, use_pred, load_path, expected_score",
    [
        (
            "mnist",
            "generate",
            "load_mnist_model",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            3,
            "load_mnist_dataset",
            10,
            1.0,
            27,
            "self-influence",
            8,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity, "cache_dir": "cache", "model_id": "test"},
            False,
            None,
            0.4921875,
        ),
        (
            "mnist",
            "assemble",
            "load_mnist_model",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            3,
            "load_mnist_dataset",
            10,
            1.0,
            27,
            SumAggregator,
            8,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity, "cache_dir": "cache", "model_id": "test"},
            False,
            None,
            0.4921875,
        ),
        (
            "mnist",
            "load",
            "load_mnist_model",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            3,
            "load_mnist_dataset",
            10,
            1.0,
            27,
            SumAggregator,
            8,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity, "cache_dir": "cache", "model_id": "test"},
            False,
            "tests/assets/mnist_mislabel_detection_state_dict",
            0.4921875,
        ),
        (
            "mnist",
            "assemble",
            "load_mnist_model",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            3,
            "load_mnist_dataset",
            10,
            1.0,
            27,
            SumAggregator,
            8,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity, "cache_dir": "cache", "model_id": "test"},
            True,
            None,
            0.4921875,
        ),
    ],
)
def test_mislabeling_detection(
    test_id,
    init_method,
    model,
    optimizer,
    lr,
    criterion,
    max_epochs,
    dataset,
    n_classes,
    p,
    seed,
    batch_size,
    global_method,
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
        trainer = Trainer(
            max_epochs=max_epochs,
            optimizer=optimizer,
            lr=lr,
            criterion=criterion,
        )

        dst_eval = MislabelingDetection.generate(
            model=model,
            trainer=trainer,
            train_dataset=dataset,
            n_classes=n_classes,
            p=p,
            global_method=global_method,
            class_to_group="random",
            trainer_fit_kwargs={"max_epochs": max_epochs},
            seed=seed,
            batch_size=batch_size,
            device="cpu",
        )

    elif init_method == "load":
        dst_eval = MislabelingDetection.load(path=load_path)
    elif init_method == "assemble":
        dst_eval = MislabelingDetection.assemble(
            model=model, train_dataset=dataset, n_classes=n_classes, p=p, global_method=global_method, batch_size=batch_size
        )
    else:
        raise ValueError(f"Invalid init_method: {init_method}")

    score = dst_eval.evaluate(
        expl_dataset=dataset,
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        cache_dir=str(tmp_path),
        model_id="default_model_id",
        use_predictions=use_pred,
        batch_size=batch_size,
        device="cpu",
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)


@pytest.mark.toy_benchmarks
@pytest.mark.parametrize(
    "test_id, pl_module, max_epochs, dataset, n_classes, p, seed, "
    "global_method, batch_size, explainer_cls, expl_kwargs, use_pred, load_path, expected_score",
    [
        (
            "mnist",
            "load_mnist_pl_module",
            3,
            "load_mnist_dataset",
            10,
            1.0,
            27,
            "self-influence",
            8,
            CaptumSimilarity,
            {"layers": "model.fc_2", "similarity_metric": cosine_similarity, "cache_dir": "cache", "model_id": "test"},
            False,
            None,
            0.4921875,
        ),
    ],
)
def test_mislabeling_detection_generate_from_pl_module(
    test_id,
    pl_module,
    max_epochs,
    dataset,
    n_classes,
    p,
    seed,
    batch_size,
    global_method,
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

    dst_eval = MislabelingDetection.generate(
        model=pl_module,
        trainer=trainer,
        train_dataset=dataset,
        n_classes=n_classes,
        p=p,
        global_method=global_method,
        class_to_group="random",
        trainer_fit_kwargs={},
        seed=seed,
        batch_size=batch_size,
        device="cpu",
    )

    score = dst_eval.evaluate(
        expl_dataset=dataset,
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        cache_dir=str(tmp_path),
        model_id="default_model_id",
        use_predictions=use_pred,
        batch_size=batch_size,
        device="cpu",
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)
