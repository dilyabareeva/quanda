"""Contains tests common to all benchmarks."""
import os
import math

import pytest

from quanda.benchmarks.downstream_eval import (
    ClassDetection,
    MislabelingDetection,
    ShortcutDetection,
    SubclassDetection,
)
from quanda.benchmarks.heuristics import (
    ModelRandomization,
    TopKCardinality,
    MixedDatasets,
)
from quanda.explainers.wrappers import CaptumSimilarity
from quanda.utils.functions import cosine_similarity


@pytest.mark.tested
@pytest.mark.parametrize(
    "test_id, config, load_from_disk, bench_cls, explainer_cls, expl_kwargs, expected_score",
    [
        (
            "mnist",
            "load_mnist_unit_test_config",
            True,
            TopKCardinality,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.614,
        ),
        (
            "mnist",
            "load_mnist_unit_test_config",
            True,
            ModelRandomization,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.30487003922462463,
        ),
        (
            "mnist",
            "load_mnist_mixed_config",
            True,
            MixedDatasets,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.017659740522503853,
        ),
        (
            "mnist",
            "load_mnist_unit_test_config",
            True,
            ClassDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.6600000262260437,
        ),
        (
            "mnist",
            "load_mnist_mislabeling_config",
            False,
            MislabelingDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.44353821873664856,
        ),
        (
            "mnist",
            "load_mnist_mislabeling_config",
            True,
            MislabelingDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.44353821873664856,
        ),
        (
            "mnist",
            "load_mnist_shortcut_config",
            True,
            ShortcutDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.12516948580741882,
        ),
        (
            "mnist",
            "load_mnist_shortcut_config",
            False,
            ShortcutDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.12516948580741882,
        ),
        (
            "mnist",
            "load_mnist_subclass_config",
            True,
            SubclassDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.23000000417232513,
        ),
    ],
)
def test_bench_from_config(
    test_id,
    config,
    load_from_disk,
    bench_cls,
    explainer_cls,
    expl_kwargs,
    expected_score,
    tmp_path,
    request,
):
    config = request.getfixturevalue(config)

    expl_kwargs = {
        **expl_kwargs,
        "model_id": "test",
        "cache_dir": str(tmp_path),
    }

    config["cache_dir"] = str(tmp_path)
    dst_eval = bench_cls.from_config(
        config=config,
        load_meta_from_disk=load_from_disk,
    )

    score = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=8,
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)


@pytest.mark.tested
@pytest.mark.parametrize(
    "test_id, config, load_from_disk, bench_cls, explainer_cls, expl_kwargs",
    [
        (
            "mnist",
            "load_mnist_unit_test_config",
            True,
            TopKCardinality,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
        ),
        (
            "mnist",
            "load_mnist_unit_test_config",
            True,
            ModelRandomization,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
        ),
        (
            "mnist",
            "load_mnist_mixed_config",
            True,
            MixedDatasets,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
        ),
        (
            "mnist",
            "load_mnist_unit_test_config",
            True,
            ClassDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
        ),
        (
            "mnist",
            "load_mnist_mislabeling_config",
            False,
            MislabelingDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
        ),
        (
            "mnist",
            "load_mnist_mislabeling_config",
            True,
            MislabelingDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
        ),
        (
            "mnist",
            "load_mnist_shortcut_config",
            True,
            ShortcutDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
        ),
        (
            "mnist",
            "load_mnist_shortcut_config",
            False,
            ShortcutDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
        ),
        (
            "mnist",
            "load_mnist_subclass_config",
            True,
            SubclassDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
        ),
    ],
)
def test_train_from_config(
    test_id,
    config,
    load_from_disk,
    bench_cls,
    explainer_cls,
    expl_kwargs,
    tmp_path,
    request,
):
    config = request.getfixturevalue(config)

    expl_kwargs = {
        **expl_kwargs,
        "model_id": "test",
        "cache_dir": str(tmp_path),
    }

    config["ckpt_dir"] = os.path.join(str(tmp_path), "ckpt")
    config["metadata_dir"] = os.path.join(str(tmp_path), "meta")

    # create the dirs
    os.makedirs(config["ckpt_dir"], exist_ok=True)
    os.makedirs(config["metadata_dir"], exist_ok=True)

    dst_eval = bench_cls.train(
        config=config,
        load_meta_from_disk=load_from_disk,
    )

    score = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=8,
    )["score"]
