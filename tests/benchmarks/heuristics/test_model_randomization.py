import math

import pytest
import torch

from quanda.benchmarks.heuristics import ModelRandomization
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
            0.5208332538604736,
        ),
        (
            "mnist2",
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
            0.5208332538604736,
        ),
    ],
)
def test_model_randomization(
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
        dst_eval = ModelRandomization.generate(
            model=model,
            cache_dir=str(tmp_path),
            checkpoints=checkpoint,
            train_dataset=dataset,
            eval_dataset=dataset,
            device="cpu",
        )

    elif init_method == "assemble":
        dst_eval = ModelRandomization.assemble(
            model=model,
            cache_dir=str(tmp_path),
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


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, benchmark, batch_size, explainer_cls, expl_kwargs, expected_score",
    [
        (
            "mnist",
            "mnist_model_randomization_benchmark",
            8,
            CaptumSimilarity,
            {
                "layers": "model.fc_2",
                "similarity_metric": cosine_similarity,
                "load_from_disk": True,
            },
            0.509926438331604,
        ),
    ],
)
def test_model_randomization_download(
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
    dst_eval.train_dataset = torch.utils.data.Subset(
        dst_eval.train_dataset, list(range(16))
    )
    dst_eval.eval_dataset = torch.utils.data.Subset(
        dst_eval.eval_dataset, list(range(16))
    )
    score = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=batch_size,
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)
