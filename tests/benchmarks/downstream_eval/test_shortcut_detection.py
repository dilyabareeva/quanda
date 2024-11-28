import math
from functools import reduce

import lightning as L
import pytest
import torch
from torcheval.metrics.functional import binary_auprc

from quanda.benchmarks.downstream_eval import ShortcutDetection
from quanda.explainers.wrappers.captum_influence import CaptumSimilarity
from quanda.utils.datasets.transformed import SampleTransformationDataset
from quanda.utils.functions.similarities import cosine_similarity
from quanda.utils.training.trainer import Trainer


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, init_method, model, checkpoint, optimizer, lr, criterion, max_epochs, dataset, sample_fn, n_classes, shortcut_cls,"
    "shortcut_indices, p, seed, batch_size, explainer_cls, expl_kwargs, filter_by_class, expected_score",
    [
        (
            "mnist_generate",
            "generate",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
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
            "load_mnist_last_checkpoint",
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
    checkpoint,
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
    checkpoint = request.getfixturevalue(checkpoint)
    optimizer = request.getfixturevalue(optimizer)
    criterion = request.getfixturevalue(criterion)
    dataset = request.getfixturevalue(dataset)
    sample_fn = request.getfixturevalue(sample_fn)
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

        dst_eval = ShortcutDetection.generate(
            model=model,
            trainer=trainer,
            base_dataset=dataset,
            n_classes=n_classes,
            eval_dataset=dataset,
            filter_by_class=filter_by_class,
            filter_by_prediction=True,
            shortcut_cls=shortcut_cls,
            p=p,
            sample_fn=sample_fn,
            trainer_fit_kwargs={"max_epochs": max_epochs},
            seed=seed,
            cache_dir=str(tmp_path),
            batch_size=batch_size,
        )
    elif init_method == "assemble":
        dst_eval = ShortcutDetection.assemble(
            model=model,
            checkpoints=checkpoint,
            base_dataset=dataset,
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
            0.32718,
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
    expl_kwargs = {
        **expl_kwargs,
        "model_id": "test",
        "cache_dir": str(tmp_path),
    }
    trainer = L.Trainer(max_epochs=max_epochs)
    sample_fn = request.getfixturevalue(sample_fn)
    dst_eval = ShortcutDetection.generate(
        model=pl_module,
        trainer=trainer,
        base_dataset=dataset,
        n_classes=n_classes,
        eval_dataset=dataset,
        filter_by_class=True,
        filter_by_prediction=False,
        shortcut_cls=shortcut_cls,
        p=p,
        sample_fn=sample_fn,
        trainer_fit_kwargs={},
        seed=seed,
        cache_dir=str(tmp_path),
        batch_size=batch_size,
    )

    results = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=batch_size,
    )
    assert math.isclose(results["score"], expected_score, abs_tol=0.00001)


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, benchmark, batch_size, explainer_cls, expl_kwargs, filter_by_class, filter_by_prediction, expected_score",
    [
        (
            "mnist",
            "mnist_shortcut_detection_benchmark",
            8,
            CaptumSimilarity,
            {
                "layers": "model.fc_2",
                "similarity_metric": cosine_similarity,
                "load_from_disk": True,
            },
            True,
            True,
            "compute",
        ),
        (
            "mnist",
            "mnist_shortcut_detection_benchmark",
            8,
            CaptumSimilarity,
            {
                "layers": "model.fc_2",
                "similarity_metric": cosine_similarity,
                "load_from_disk": True,
            },
            True,
            False,
            "compute",
        ),
        (
            "mnist",
            "mnist_shortcut_detection_benchmark",
            8,
            CaptumSimilarity,
            {
                "layers": "model.fc_2",
                "similarity_metric": cosine_similarity,
                "load_from_disk": True,
            },
            False,
            True,
            "compute",
        ),
        (
            "mnist",
            "mnist_shortcut_detection_benchmark",
            8,
            CaptumSimilarity,
            {
                "layers": "model.fc_2",
                "similarity_metric": cosine_similarity,
                "load_from_disk": True,
            },
            False,
            False,
            "compute",
        ),
    ],
)
def test_shortcut_detection_download(
    test_id,
    benchmark,
    batch_size,
    explainer_cls,
    expl_kwargs,
    filter_by_class,
    filter_by_prediction,
    expected_score,
    tmp_path,
    request,
):
    dst_eval = request.getfixturevalue(benchmark)

    expl_kwargs = {"model_id": "0", "cache_dir": str(tmp_path), **expl_kwargs}
    dst_eval.base_dataset = torch.utils.data.Subset(
        dst_eval.base_dataset, list(range(16))
    )
    dst_eval.shortcut_dataset = torch.utils.data.Subset(
        dst_eval.shortcut_dataset, list(range(16))
    )
    dst_eval.eval_dataset = torch.utils.data.Subset(
        dst_eval.eval_dataset, list(range(16))
    )
    dst_eval.shortcut_indices = [
        i for i in dst_eval.shortcut_indices if i < 16
    ]
    dst_eval.filter_by_class = filter_by_class
    dst_eval.filter_by_prediction = filter_by_prediction

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
        shortcut_expl_ds = SampleTransformationDataset(
            dataset=dst_eval.eval_dataset,
            dataset_transform=dst_eval.dataset_transform,
            n_classes=dst_eval.n_classes,
            sample_fn=dst_eval.sample_fn,
            p=1.0,
        )
        train_ld = torch.utils.data.DataLoader(
            dst_eval.shortcut_dataset, batch_size=16, shuffle=False
        )
        test_ld = torch.utils.data.DataLoader(
            shortcut_expl_ds, batch_size=16, shuffle=False
        )
        for x, y in iter(train_ld):
            x = x.to(dst_eval.device)
            dst_eval.model(x)
        act_train = activation[0]
        activation = []
        for x, y in iter(test_ld):
            x = x.to(dst_eval.device)
            y_test = y.to(dst_eval.device)
            y_preds = dst_eval.model(x).argmax(dim=-1)
            select_idx = torch.tensor([True] * 16)
            if filter_by_class:
                select_idx *= y_test != dst_eval.shortcut_cls
            if filter_by_prediction:
                select_idx *= y_preds == dst_eval.shortcut_cls
            dst_eval.model(x)
        act_test = activation[0]
        act_test = act_test[select_idx]
        act_test = torch.nn.functional.normalize(act_test, dim=-1)
        act_train = torch.nn.functional.normalize(act_train, dim=-1)
        IP = torch.matmul(act_test, act_train.T)
        binary_shortcut_indices: torch.Tensor = torch.tensor(
            [1 if i in dst_eval.shortcut_indices else 0 for i in range(16)],
            device=dst_eval.device,
        )
        expected_score = (
            torch.tensor(
                [binary_auprc(xpl, binary_shortcut_indices) for xpl in IP]
            )
            .mean()
            .item()
        )

    assert math.isclose(score, expected_score, abs_tol=0.00001)


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, benchmark",
    [
        (
            "mnist",
            "mnist_shortcut_detection_benchmark",
        ),
    ],
)
def test_shortcut_detection_download_sanity_checks(
    test_id, benchmark, request
):
    dst_eval = request.getfixturevalue(benchmark)
    assertions = []
    for i in dst_eval.shortcut_indices:
        x, y = dst_eval.shortcut_dataset[i]
        assertions.append(y == dst_eval.shortcut_cls)
        assertions.append(torch.allclose(x[0, 15:23, 15:23], torch.ones(8, 8)))
    assert all(assertions)
