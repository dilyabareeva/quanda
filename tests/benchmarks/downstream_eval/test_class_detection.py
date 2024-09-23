import math

import datasets
import pytest
import torch.utils.data

from quanda.benchmarks.downstream_eval import ClassDetection
from quanda.explainers.wrappers import CaptumSimilarity
from quanda.utils.functions import cosine_similarity


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, init_method, model, dataset, n_classes, n_groups, seed, test_labels, "
    "batch_size, explainer_cls, expl_kwargs, load_path, expected_score",
    [
        (
            "mnist",
            "generate",
            "load_mnist_model",
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
    dataset = request.getfixturevalue(dataset)

    if init_method == "generate":
        dst_eval = ClassDetection.generate(
            model=model,
            train_dataset=dataset,
            eval_dataset=dataset,
            device="cpu",
        )

    elif init_method == "assemble":
        dst_eval = ClassDetection.assemble(model=model, train_dataset=dataset, eval_dataset=dataset)
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
    "test_id, init_method, model, dataset, dataset_split, n_classes, n_groups, seed, test_labels, "
    "batch_size, explainer_cls, expl_kwargs, load_path, expected_score",
    [
        (
            "mnist",
            "generate",
            "load_mnist_model",
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

    if init_method == "generate":
        dst_eval = ClassDetection.generate(
            model=model,
            train_dataset=dataset,
            eval_dataset=dataset,
            dataset_split=dataset_split,
            device="cpu",
        )

    elif init_method == "assemble":
        dst_eval = ClassDetection.assemble(
            model=model,
            train_dataset=dataset,
            eval_dataset=dataset,
            dataset_split=dataset_split,
        )
    else:
        raise ValueError(f"Invalid init_method: {init_method}")

    assert isinstance(dst_eval.train_dataset, datasets.arrow_dataset.Dataset)


@pytest.mark.tested
@pytest.mark.parametrize(
    "test_id, benchmark_name, batch_size, explainer_cls, expl_kwargs, expected_score",
    [
        (
            "mnist",
            "mnist_class_detection",
            8,
            CaptumSimilarity,
            {
                "layers": "model.fc_2",
                "similarity_metric": cosine_similarity,
                "load_from_disk": True,
            },
            0.9375,
        ),
    ],
)
def test_class_detection_download(
    test_id,
    benchmark_name,
    batch_size,
    explainer_cls,
    expl_kwargs,
    expected_score,
    tmp_path,
):


    dst_eval = ClassDetection.download(
        name=benchmark_name,
        cache_dir=str(tmp_path),
        device="cpu",
    )

    expl_kwargs = {"model_id": "0", "cache_dir": str(tmp_path), **expl_kwargs}
    dst_eval.train_dataset = torch.utils.data.Subset(dst_eval.eval_dataset, list(range(16)))
    dst_eval.eval_dataset = torch.utils.data.Subset(dst_eval.eval_dataset, list(range(16)))
    score = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=batch_size,
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)