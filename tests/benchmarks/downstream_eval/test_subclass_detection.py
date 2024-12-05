import math
from functools import reduce

import lightning as L
import pytest
import torch

from quanda.benchmarks.downstream_eval.subclass_detection import (
    SubclassDetection,
)
from quanda.explainers.wrappers.captum_influence import CaptumSimilarity
from quanda.utils.functions.similarities import cosine_similarity
from quanda.utils.training.trainer import Trainer


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, init_method, model, checkpoint, optimizer, lr, criterion, max_epochs, dataset, n_classes, n_groups, seed, "
    "class_to_group, batch_size, explainer_cls, expl_kwargs, use_pred, load_path, expected_score",
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
            2,
            27,
            {i: i % 2 for i in range(10)},
            8,
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            False,
            None,
            1.0,
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
            2,
            27,
            {i: i % 2 for i in range(10)},
            8,
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            False,
            None,
            1.0,
        ),
    ],
)
def test_subclass_detection(
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
    n_groups,
    seed,
    class_to_group,
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
    checkpoint = request.getfixturevalue(checkpoint)
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

        dst_eval = SubclassDetection.generate(
            model=model,
            trainer=trainer,
            base_dataset=dataset,
            eval_dataset=dataset,
            n_classes=n_classes,
            n_groups=n_groups,
            class_to_group=class_to_group,
            trainer_fit_kwargs={"max_epochs": max_epochs},
            seed=seed,
            batch_size=batch_size,
            cache_dir=str(tmp_path),
            device="cpu",
        )

    elif init_method == "assemble":
        dst_eval = SubclassDetection.assemble(
            group_model=model,
            checkpoints=checkpoint,
            base_dataset=dataset,
            eval_dataset=dataset,
            n_classes=n_classes,
            class_to_group=class_to_group,
        )
    else:
        raise ValueError(f"Invalid init_method: {init_method}")

    expl_kwargs = {"model_id": "0", "cache_dir": str(tmp_path), **expl_kwargs}
    score = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=batch_size,
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, pl_module, max_epochs, dataset, n_classes, n_groups, seed, "
    "class_to_group, batch_size, explainer_cls, expl_kwargs, use_pred, load_path, expected_score",
    [
        (
            "mnist",
            "load_mnist_pl_module",
            3,
            "load_mnist_dataset",
            10,
            2,
            27,
            {i: i % 2 for i in range(10)},
            8,
            CaptumSimilarity,
            {
                "layers": "model.fc_2",
                "similarity_metric": cosine_similarity,
            },
            False,
            None,
            1.0,
        ),
    ],
)
def test_subclass_detection_generate_lightning_model(
    test_id,
    pl_module,
    max_epochs,
    dataset,
    n_classes,
    n_groups,
    seed,
    class_to_group,
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

    dst_eval = SubclassDetection.generate(
        model=pl_module,
        trainer=trainer,
        base_dataset=dataset,
        eval_dataset=dataset,
        n_classes=n_classes,
        n_groups=n_groups,
        class_to_group=class_to_group,
        trainer_fit_kwargs={},
        seed=seed,
        cache_dir=str(tmp_path),
        batch_size=batch_size,
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
            "mnist_subclass_detection_benchmark",
            8,
            CaptumSimilarity,
            {
                "layers": "model.fc_2",
                "similarity_metric": cosine_similarity,
                "load_from_disk": True,
            },
            "compute",
        ),
    ],
)
def test_subclass_detection_download(
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
    dst_eval.grouped_dataset = torch.utils.data.Subset(
        dst_eval.grouped_dataset, list(range(16))
    )
    dst_eval.eval_dataset = torch.utils.data.Subset(
        dst_eval.eval_dataset, list(range(16))
    )
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
            getattr, expl_kwargs["layers"].split("."), dst_eval.group_model
        )
        exp_layer.register_forward_hook(hook)
        train_ld = torch.utils.data.DataLoader(
            dst_eval.grouped_dataset, batch_size=16, shuffle=False
        )
        test_ld = torch.utils.data.DataLoader(
            dst_eval.eval_dataset, batch_size=16, shuffle=False
        )
        for x, y in iter(train_ld):
            x = x.to(dst_eval.device)
            dst_eval.group_model(x)
        act_train = activation[0]
        activation = []
        y_train = torch.tensor([y for x, y in dst_eval.base_dataset])
        for x, y in iter(test_ld):
            x = x.to(dst_eval.device)
            y_test = y.to(dst_eval.device)
            dst_eval.group_model(x)
        act_test = activation[0]
        act_test = torch.nn.functional.normalize(act_test, dim=-1)
        act_train = torch.nn.functional.normalize(act_train, dim=-1)
        IP = torch.matmul(act_test, act_train.T)
        max_attr_indices = IP.argmax(dim=-1)
        expected_score = (
            torch.sum((y_train[max_attr_indices] == y_test) * 1.0)
            / act_test.shape[0]
        )

    assert math.isclose(score, expected_score, abs_tol=0.00001)


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, benchmark, batch_size",
    [
        (
            "mnist",
            "mnist_subclass_detection_benchmark",
            8,
        ),
    ],
)
def test_subclass_detection_download_sanity_checks(
    test_id, benchmark, batch_size, request
):
    dst_eval = request.getfixturevalue(benchmark)
    assertions = [
        dst_eval.grouped_dataset[i][1]
        == dst_eval.class_to_group[dst_eval.base_dataset[i][1]]
        for i in range(len(dst_eval.grouped_dataset))
    ]
    assert all(assertions)
