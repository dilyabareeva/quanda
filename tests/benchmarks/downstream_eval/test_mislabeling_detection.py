import math
from functools import reduce

import lightning as L
import pytest
import torch

from quanda.benchmarks.downstream_eval.mislabeling_detection import (
    MislabelingDetection,
)
from quanda.explainers.global_ranking.aggregators import SumAggregator
from quanda.explainers.wrappers.captum_influence import CaptumSimilarity
from quanda.utils.functions.similarities import (
    cosine_similarity,
    dot_product_similarity,
)
from quanda.utils.training.trainer import Trainer


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, init_method, model, checkpoint, optimizer, lr, criterion, max_epochs, dataset, n_classes, p, seed, "
    "global_method, mislabeling_labels, batch_size, explainer_cls, expl_kwargs, use_pred, load_path, expected_score",
    [
        (
            "mnist",
            "generate",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            3,
            "load_mnist_dataset",
            10,
            1.0,
            27,
            "self-influence",
            "mnist_seed_27_mislabeling_labels",
            8,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            False,
            None,
            0.4921875,
        ),
        (
            "mnist",
            "assemble",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            3,
            "load_mnist_dataset",
            10,
            1.0,
            27,
            SumAggregator,
            "mnist_seed_27_mislabeling_labels",
            8,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            False,
            None,
            0.0,
        ),
        (
            "mnist",
            "assemble",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            3,
            "load_mnist_dataset",
            10,
            1.0,
            27,
            SumAggregator,
            "mnist_seed_27_mislabeling_labels",
            8,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            True,
            None,
            0.0,
        ),
    ],
)
def test_mislabeling_detection(
    test_id,
    init_method,
    model,
    checkpoint,
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
    mislabeling_labels,
    explainer_cls,
    expl_kwargs,
    use_pred,
    load_path,
    expected_score,
    tmp_path,
    request,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    optimizer = request.getfixturevalue(optimizer)
    criterion = request.getfixturevalue(criterion)
    dataset = request.getfixturevalue(dataset)
    mislabeling_labels = request.getfixturevalue(mislabeling_labels)

    expl_kwargs = {
        **expl_kwargs,
        "model_id": "test",
        "cache_dir": str(tmp_path),
    }
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
            base_dataset=dataset,
            eval_dataset=dataset,
            n_classes=n_classes,
            p=p,
            global_method=global_method,
            trainer_fit_kwargs={"max_epochs": max_epochs},
            seed=seed,
            batch_size=batch_size,
            cache_dir=str(tmp_path),
            device="cpu",
        )

    elif init_method == "assemble":
        dst_eval = MislabelingDetection.assemble(
            model=model,
            checkpoints=checkpoint,
            base_dataset=dataset,
            eval_dataset=dataset,
            n_classes=n_classes,
            p=p,
            mislabeling_labels=mislabeling_labels,
            global_method=global_method,
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"Invalid init_method: {init_method}")

    score = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=batch_size,
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)


@pytest.mark.benchmarks
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
            {"layers": "model.fc_2", "similarity_metric": cosine_similarity},
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
    expl_kwargs = {
        **expl_kwargs,
        "model_id": "test",
        "cache_dir": str(tmp_path),
    }
    trainer = L.Trainer(max_epochs=max_epochs)

    dst_eval = MislabelingDetection.generate(
        model=pl_module,
        trainer=trainer,
        base_dataset=dataset,
        n_classes=n_classes,
        eval_dataset=dataset,
        p=p,
        global_method=global_method,
        class_to_group="random",
        trainer_fit_kwargs={},
        seed=seed,
        batch_size=batch_size,
        cache_dir=str(tmp_path),
        device="cpu",
    )

    expl_kwargs = {"model_id": "0", "cache_dir": str(tmp_path), **expl_kwargs}
    score = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=batch_size,
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, benchmark, batch_size, explainer_cls, expl_kwargs, expected_score",
    [
        (
            "mnist",
            "mnist_mislabeling_detection_benchmark",
            8,
            CaptumSimilarity,
            {
                "layers": "model.fc_2",
                "similarity_metric": dot_product_similarity,
                "load_from_disk": True,
            },
            "compute",
        ),
    ],
)
def test_mislabeling_detection_download(
    test_id,
    benchmark,
    batch_size,
    explainer_cls,
    expl_kwargs,
    expected_score,
    tmp_path,
    request,
):
    dst_eval = request.getfixturevalue(benchmark)

    expl_kwargs = {"model_id": "0", "cache_dir": str(tmp_path), **expl_kwargs}
    dst_eval.base_dataset = torch.utils.data.Subset(
        dst_eval.base_dataset, list(range(16))
    )
    dst_eval.mislabeling_dataset = torch.utils.data.Subset(
        dst_eval.mislabeling_dataset, list(range(16))
    )
    dst_eval.mislabeling_labels = {
        key: val
        for key, val in dst_eval.mislabeling_labels.items()
        if key in list(range(16))
    }
    dst_eval.mislabeling_indices = [
        key for key in dst_eval.mislabeling_labels.keys()
    ]

    score = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=batch_size,
    )["score"]
    if expected_score == "compute":
        activation = []

        def hook(model, input, output):
            activation.append(output.detach())

        exp_layer = reduce(
            getattr, expl_kwargs["layers"].split("."), dst_eval.model
        )
        exp_layer.register_forward_hook(hook)
        train_ld = torch.utils.data.DataLoader(
            dst_eval.mislabeling_dataset, batch_size=16, shuffle=False
        )
        for x, y in iter(train_ld):
            x = x.to(dst_eval.device)
            dst_eval.model(x)
        act_train = activation[0]
        activation = []
        IP = torch.matmul(act_train, act_train.T)
        global_ranking = IP.diag().argsort()
        detection_curve = torch.tensor(
            [
                1.0 if i.item() in dst_eval.mislabeling_labels.keys() else 0.0
                for i in global_ranking
            ]
        )
        expected_score = torch.trapezoid(
            torch.cumsum(detection_curve, dim=0)
        ) / (len(dst_eval.mislabeling_indices) * 16)
    assert math.isclose(score, expected_score, abs_tol=0.0001)


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, benchmark",
    [
        (
            "mnist",
            "mnist_mislabeling_detection_benchmark",
        ),
    ],
)
def test_mislabeling_detection_download_sanity_checks(
    test_id, benchmark, tmp_path, request
):
    dst_eval = request.getfixturevalue(benchmark)
    assertions = []
    for ind, label in list(dst_eval.mislabeling_labels.items())[0:500]:
        assertions.append(
            (dst_eval.base_dataset[ind][1] != label)
            and (dst_eval.mislabeling_dataset[ind][1] == label)
        )
    assert all(assertions)
