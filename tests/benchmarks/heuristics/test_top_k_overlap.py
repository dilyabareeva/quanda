import math

import pytest
import torch

from quanda.benchmarks.heuristics import TopKCardinality
from quanda.explainers.wrappers import CaptumSimilarity
from quanda.utils.functions import cosine_similarity


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, init_method, model, checkpoint, dataset, n_classes, n_groups, seed, test_labels, "
    "batch_size, use_predictions, explainer_cls, expl_kwargs, load_path, expected_score",
    [
        (
            "mnist1",
            "generate",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            10,
            2,
            27,
            "load_mnist_test_labels_1",
            8,
            False,
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            None,
            1.0,
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
            False,
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
def test_topk_cardinality(
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
    use_predictions,
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
        dst_eval = TopKCardinality.generate(
            model=model,
            checkpoints=checkpoint,
            train_dataset=dataset,
            eval_dataset=dataset,
            device="cpu",
        )

    elif init_method == "assemble":
        dst_eval = TopKCardinality.assemble(
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


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, benchmark_name, batch_size, explainer_cls, expl_kwargs, expected_score",
    [
        (
            "mnist",
            "mnist_top_k_cardinality_benchmark",
            8,
            CaptumSimilarity,
            {
                "layers": "model.fc_2",
                "similarity_metric": cosine_similarity,
                "load_from_disk": True,
            },
            0.5625,
        ),
    ],
)
def test_top_k_cardinality_download(
    test_id,
    benchmark_name,
    batch_size,
    explainer_cls,
    expl_kwargs,
    expected_score,
    tmp_path,
    request,
):
    dst_eval = request.getfixturevalue(benchmark_name)

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
