import math

import pytest
import torch

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


@pytest.mark.production_bench
@pytest.mark.parametrize(
    "config_name",
    [
        "mnist_shortcut_detection_unit",
        "mnist_shortcut_detection",
    ],
)
def test_shortcut_transform_indices(config_name, tmp_path):
    """Verify shortcut-transformed eval samples stay unchanged
    after re-applying the sample transform."""
    bench = ShortcutDetection.load_pretrained(
        config_name=config_name,
        cache_dir=str(tmp_path),
        load_meta_from_disk=True,
    )

    eval_dl = torch.utils.data.DataLoader(
        torch.utils.data.Subset(
            bench.eval_dataset,
            bench.eval_dataset.transform_indices,
        ),
        batch_size=32,
        shuffle=False,
    )

    total, unchanged = 0, 0
    for x, _ in eval_dl:
        x_re = torch.stack(
            [bench.eval_dataset.sample_fn(xi) for xi in x]
        )
        unchanged += torch.all(
            (x == x_re).view(x.size(0), -1), dim=1
        ).sum().item()
        total += x.size(0)

    pct = unchanged / total
    assert pct > 0, (
        f"Expected some unchanged shortcut samples for "
        f"{config_name}, got {pct}"
    )
