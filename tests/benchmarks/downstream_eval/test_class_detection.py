import math
from functools import reduce

import datasets
import pytest
import torch.utils.data

from quanda.benchmarks.downstream_eval import ClassDetection
from quanda.explainers.wrappers import CaptumSimilarity
from quanda.utils.functions import cosine_similarity


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, init_method, model, checkpoint, dataset, n_classes, n_groups, seed, test_labels, "
    "batch_size, explainer_cls, expl_kwargs, load_path, expected_score",
    [
        (
            "mnist",
            "generate",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            10,
            2,
            27,
            "load_mnist_test_labels_1",
            8,
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            None,
            1.0,
        ),
        (
            "mnist",
            "assemble",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            10,
            2,
            27,
            "load_mnist_test_labels_1",
            8,
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            None,
            1.0,
        ),
    ],
)
def test_class_detection(
    test_id,
    init_method,
    model,
    checkpoint,
    dataset,
    n_classes,
    n_groups,
    seed,
    test_labels,
    batch_size,
    explainer_cls,
    expl_kwargs,
    load_path,
    expected_score,
    tmp_path,
    request,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    dataset = request.getfixturevalue(dataset)

    if init_method == "generate":
        dst_eval = ClassDetection.generate(
            model=model,
            checkpoints=checkpoint,
            train_dataset=dataset,
            eval_dataset=dataset,
            device="cpu",
        )

    elif init_method == "assemble":
        dst_eval = ClassDetection.assemble(
            model=model,
            checkpoints=checkpoint,
            train_dataset=dataset,
            eval_dataset=dataset,
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


@pytest.mark.local
@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, init_method, model, checkpoint, dataset, dataset_split, n_classes, n_groups, seed, test_labels, "
    "batch_size, explainer_cls, expl_kwargs, load_path, expected_score",
    [
        (
            "mnist",
            "generate",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "imdb",
            "train[:1%]",
            10,
            2,
            27,
            "load_mnist_test_labels_1",
            8,
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            None,
            1.0,
        ),
        (
            "mnist",
            "assemble",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "imdb",
            "train[:1%]",
            10,
            2,
            27,
            "load_mnist_test_labels_1",
            8,
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            None,
            1.0,
        ),
    ],
)
def test_class_detection_hugging_face(
    test_id,
    init_method,
    model,
    checkpoint,
    dataset,
    dataset_split,
    n_classes,
    n_groups,
    seed,
    test_labels,
    batch_size,
    explainer_cls,
    expl_kwargs,
    load_path,
    expected_score,
    tmp_path,
    request,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)

    if init_method == "generate":
        dst_eval = ClassDetection.generate(
            model=model,
            checkpoints=checkpoint,
            train_dataset=dataset,
            eval_dataset=dataset,
            dataset_split=dataset_split,
            device="cpu",
        )

    elif init_method == "assemble":
        dst_eval = ClassDetection.assemble(
            model=model,
            checkpoints=checkpoint,
            train_dataset=dataset,
            eval_dataset=dataset,
            dataset_split=dataset_split,
        )
    else:
        raise ValueError(f"Invalid init_method: {init_method}")

    assert isinstance(
        dst_eval.train_dataset.dataset, datasets.arrow_dataset.Dataset
    )


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, benchmark, batch_size, explainer_cls, expl_kwargs, use_predictions, expected_score",
    [
        (
            "mnist",
            "mnist_class_detection_benchmark",
            8,
            CaptumSimilarity,
            {
                "layers": "model.fc_2",
                "similarity_metric": cosine_similarity,
                "load_from_disk": True,
            },
            True,
            "compute",
        ),
        (
            "mnist",
            "mnist_class_detection_benchmark",
            8,
            CaptumSimilarity,
            {
                "layers": "model.fc_2",
                "similarity_metric": cosine_similarity,
                "load_from_disk": True,
            },
            False,
            "compute",
        ),
    ],
)
def test_class_detection_download(
    test_id,
    benchmark,
    batch_size,
    explainer_cls,
    expl_kwargs,
    use_predictions,
    expected_score,
    tmp_path,
    request,
):
    dst_eval = request.getfixturevalue(benchmark)
    expl_kwargs = {"model_id": "0", "cache_dir": str(tmp_path), **expl_kwargs}
    dst_eval.train_dataset = torch.utils.data.Subset(
        dst_eval.train_dataset, list(range(16))
    )
    dst_eval.eval_dataset = torch.utils.data.Subset(
        dst_eval.eval_dataset, list(range(16))
    )
    dst_eval.use_predictions = use_predictions
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
            dst_eval.train_dataset, batch_size=16, shuffle=False
        )
        test_ld = torch.utils.data.DataLoader(
            dst_eval.eval_dataset, batch_size=16, shuffle=False
        )
        for x, y in iter(train_ld):
            x = x.to(dst_eval.device)
            y_train = y.to(dst_eval.device)
            dst_eval.model(x)
        act_train = activation[0]
        activation = []
        for x, y in iter(test_ld):
            x = x.to(dst_eval.device)
            y_test = y.to(dst_eval.device)
            y_preds = dst_eval.model(x).argmax(dim=-1)
            dst_eval.model(x)
        act_test = activation[0]
        act_test = torch.nn.functional.normalize(act_test, dim=-1)
        act_train = torch.nn.functional.normalize(act_train, dim=-1)
        IP = torch.matmul(act_test, act_train.T)
        max_attr_indices = IP.argmax(dim=-1)
        y_test_against = y_preds if use_predictions else y_test
        expected_score = (
            torch.sum((y_train[max_attr_indices] == y_test_against) * 1.0)
            / act_test.shape[0]
        )
    assert math.isclose(score, expected_score, abs_tol=0.00001)
