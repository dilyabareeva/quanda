import math
from functools import reduce

import datasets
import pytest
import torch.utils.data

from quanda.benchmarks.downstream_eval import ClassDetection
from quanda.explainers.wrappers import CaptumSimilarity
from quanda.utils.functions import cosine_similarity


@pytest.mark.tested
@pytest.mark.parametrize(
    "test_id, config, explainer_cls, expl_kwargs, expected_score",
    [
        (
            "mnist",
            "load_mnist_unit_test_config",
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.9800000190734863,
        ),
    ],
)
def test_class_detection(
    test_id, config, explainer_cls, expl_kwargs, expected_score, tmp_path, request
):
    config = request.getfixturevalue(config)

    expl_kwargs = {
        **expl_kwargs,
        "model_id": "test",
        "cache_dir": str(tmp_path),
    }

    dst_eval = ClassDetection.from_config(
        config=config,
        cache_dir=str(tmp_path),
    )

    score = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=8,
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)

