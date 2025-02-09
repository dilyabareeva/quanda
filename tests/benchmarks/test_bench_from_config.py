import math

import pytest
import torch

from quanda.benchmarks.downstream_eval import ClassDetection, \
    MislabelingDetection, ShortcutDetection, SubclassDetection
from quanda.benchmarks.heuristics import ModelRandomization, TopKCardinality, \
    MixedDatasets
from quanda.explainers.wrappers import CaptumSimilarity
from quanda.utils.functions import cosine_similarity


@pytest.mark.tested
@pytest.mark.parametrize(
    "test_id, config, bench_cls, explainer_cls, expl_kwargs, expected_score",
    [
        (
            "mnist",
            "load_mnist_unit_test_config",
            TopKCardinality,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.602,
        ),
        (
                "mnist",
                "load_mnist_unit_test_config",
                ModelRandomization,
                CaptumSimilarity,
                {"layers": "fc_2", "similarity_metric": cosine_similarity},
                0.24389608204364777,
        ),
        (
                "mnist",
                "load_mnist_mixed_config",
                MixedDatasets,
                CaptumSimilarity,
                {"layers": "fc_2", "similarity_metric": cosine_similarity},
                0.00918253418058157,
        ),
        (
                "mnist",
                "load_mnist_unit_test_config",
                ClassDetection,
                CaptumSimilarity,
                {"layers": "fc_2", "similarity_metric": cosine_similarity},
                0.9800000190734863,
        ),
        (
                "mnist",
                "load_mnist_mislabeling_config",
                MislabelingDetection,
                CaptumSimilarity,
                {"layers": "fc_2", "similarity_metric": cosine_similarity},
                0.4806745946407318,
        ),
        (
                "mnist",
                "load_mnist_shortcut_config",
                ShortcutDetection,
                CaptumSimilarity,
                {"layers": "fc_2", "similarity_metric": cosine_similarity},
                0.1492285281419754,
        ),
        (
                "mnist",
                "load_mnist_subclass_config",
                SubclassDetection,
                CaptumSimilarity,
                {"layers": "fc_2", "similarity_metric": cosine_similarity},
                0.2199999988079071,
        ),
    ],
)
def test_bench_from_config(
    test_id, config, bench_cls, explainer_cls, expl_kwargs, expected_score, tmp_path, request
):
    config = request.getfixturevalue(config)

    expl_kwargs = {
        **expl_kwargs,
        "model_id": "test",
        "cache_dir": str(tmp_path),
    }

    dst_eval = bench_cls.from_config(
        config=config,
        cache_dir=str(tmp_path),
    )

    score = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=8,
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)

