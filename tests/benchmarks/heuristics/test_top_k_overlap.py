import math

import pytest

from quanda.benchmarks.heuristics import TopKOverlap
from quanda.explainers.wrappers import CaptumSimilarity
from quanda.utils.functions import cosine_similarity


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, init_method, model, dataset, n_classes, n_groups, seed, test_labels, "
    "batch_size, use_predictions, explainer_cls, expl_kwargs, load_path, expected_score",
    [
        (
            "mnist1",
            "generate",
            "load_mnist_model",
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
    use_predictions,
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
        dst_eval = TopKOverlap.generate(
            model=model,
            train_dataset=dataset,
            device="cpu",
        )

    elif init_method == "assemble":
        dst_eval = TopKOverlap.assemble(
            model=model,
            train_dataset=dataset,
        )
    else:
        raise ValueError(f"Invalid init_method: {init_method}")

    score = dst_eval.evaluate(
        expl_dataset=dataset,
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        use_predictions=use_predictions,
        cache_dir=str(tmp_path),
        model_id="default_model_id",
        batch_size=batch_size,
        device="cpu",
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)