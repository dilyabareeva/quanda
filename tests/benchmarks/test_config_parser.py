"""Contains tests for parsing Hydra/yaml benchmark configs."""

import math

import pytest
import torch
from pandas.core.dtypes.inference import is_float

from quanda.benchmarks.config_parser import BenchConfigParser


@pytest.mark.tested
@pytest.mark.parametrize(
    "test_id, config, input_shape, offline",
    [
        (
            "mnist",
            "load_mnist_unit_test_config",
            (1, 28, 28),
            True,
        ),
        (
            "mnist",
            "load_mnist_unit_test_config",
            (1, 28, 28),
            False,
        ),
    ],
)
def test_load_ckpt_from_hf(
    test_id,
    config,
    input_shape,
    offline,
    tmp_path,
    request,
):
    config = request.getfixturevalue(config)

    model, ckpt, load_fn = BenchConfigParser.parse_model_cfg(config["model"], config["ckpt_dir"], config["id"], offline, "cpu")
    rand_input = torch.rand(1, *input_shape)
    load_fn(model, ckpt[-1])
    out = model(rand_input).mean()

    assert is_float(out.item())