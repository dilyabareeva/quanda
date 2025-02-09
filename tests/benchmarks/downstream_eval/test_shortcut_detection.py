import math
from functools import reduce

import lightning as L
import pytest
import torch

from quanda.benchmarks.downstream_eval import ShortcutDetection
from quanda.benchmarks.downstream_eval.subclass_detection import (
    SubclassDetection,
)
from quanda.explainers.wrappers.captum_influence import CaptumSimilarity
from quanda.utils.functions.similarities import cosine_similarity
from quanda.utils.training.trainer import Trainer


@pytest.mark.tested
@pytest.mark.parametrize(
    "test_id, config, explainer_cls, expl_kwargs, use_predictions, filter_by_predictions, expected_score",
    [
        (
            "mnist",
            "load_mnist_shortcut_config",
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            True,
            False,
            0.1492285281419754,
        ),
    ],
)
def test_shortcut_detection(
    test_id, config, explainer_cls, expl_kwargs,
        use_predictions, filter_by_predictions, expected_score, tmp_path, request
):
    config = request.getfixturevalue(config)

    expl_kwargs = {
        **expl_kwargs,
        "model_id": "test",
        "cache_dir": str(tmp_path),
    }

    dst_eval = ShortcutDetection.from_config(
        config=config,
        cache_dir=str(tmp_path),
    )

    score = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=8,
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)