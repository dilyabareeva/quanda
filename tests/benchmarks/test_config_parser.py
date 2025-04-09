"""Contains tests for parsing Hydra/yaml benchmark configs."""

import math

import pytest
import torch

from quanda.benchmarks.config_parser import BenchConfigParser


@pytest.mark.tested
@pytest.mark.parametrize(
    "test_id, config, input_shape",
    [
        (
            "mnist",
            "load_mnist_unit_test_config",
            (1, 28, 28),
        ),
    ],
)
def test_load_ckpt_from_hf(
    test_id,
    config,
    input_shape,
    tmp_path,
    request,
):
    config = request.getfixturevalue(config)

    rand_input = torch.rand(1, *input_shape)

    model, ckpt, load_fn = BenchConfigParser.parse_model_cfg(
        config["model"],
        str(tmp_path),
        config["repo_id"],
        config["ckpts"],
        False,
        "cpu",
    )
    load_fn(model, ckpt[-1])
    out_offline = model(rand_input).mean().item()

    model, ckpt, load_fn = BenchConfigParser.parse_model_cfg(
        config["model"],
        str(tmp_path),
        config["repo_id"],
        config["ckpts"],
        True,
        "cpu",
    )
    load_fn(model, ckpt[-1])
    out_online = model(rand_input).mean().item()

    assert math.isclose(out_offline, out_online, rel_tol=1e-5)
