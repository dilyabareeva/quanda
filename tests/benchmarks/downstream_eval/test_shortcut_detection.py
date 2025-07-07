import math

import pytest

from quanda.benchmarks.downstream_eval import (
    ShortcutDetection,
)
from quanda.explainers.wrappers import CaptumSimilarity
from quanda.utils.functions import cosine_similarity


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, config, load_from_disk, explainer_cls, expl_kwargs, expected_score",
    [
        (
            "mnist-shortcut-download",
            "load_mnist_unit_test_config",
            False,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            ValueError,
        ),
    ],
)
def test_shortcut_detection(
    test_id,
    config,
    load_from_disk,
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

    config["cache_dir"] = "bench_out"

    if isinstance(expected_score, type):
        with pytest.raises(expected_score):
            dst_eval = ShortcutDetection.from_config(
                config=config,
                load_meta_from_disk=load_from_disk,
            )
        return

    dst_eval = ShortcutDetection.from_config(
        config=config,
        load_meta_from_disk=load_from_disk,
    )

    score = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=8,
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)
