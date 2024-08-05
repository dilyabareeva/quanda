import pytest

from quanda.explainers.wrappers.captum_influence import CaptumSimilarity
from quanda.toy_benchmarks.localization.class_detection import ClassDetection
from quanda.utils.functions.similarities import cosine_similarity


@pytest.mark.toy_benchmarks
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
        (
            "mnist",
            "load",
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
            "tests/assets/mnist_class_detection_state_dict",
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
            device="cpu",
        )

    elif init_method == "load":
        dst_eval = ClassDetection.load(path=load_path)

    elif init_method == "assemble":
        dst_eval = ClassDetection.assemble(
            model=model,
            train_dataset=dataset,
        )
    else:
        raise ValueError(f"Invalid init_method: {init_method}")

    score = dst_eval.evaluate(
        expl_dataset=dataset,
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        cache_dir=str(tmp_path),
        model_id="default_model_id",
        batch_size=batch_size,
        device="cpu",
    )

    assert score == expected_score