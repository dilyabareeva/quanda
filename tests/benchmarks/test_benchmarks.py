"""Contains tests common to all benchmarks."""

import os
import math

import pytest
from omegaconf import OmegaConf

from quanda.benchmarks.config_parser import BenchConfigParser
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
    "test_id, config, load_from_disk, offline, bench_cls, explainer_cls, expl_kwargs, expected_score",
    [
        (
            "mnist",
            "load_mnist_unit_test_config",
            True,
            True,
            TopKCardinality,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.618,
        ),
        (
            "mnist",
            "load_mnist_unit_test_config",
            True,
            True,
            ModelRandomization,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.3020453453063965,
        ),
        (
            "mnist",
            "load_mnist_mixed_config",
            True,
            True,
            MixedDatasets,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.03915480896830559,
        ),
        (
            "mnist",
            "load_mnist_unit_test_config",
            True,
            True,
            ClassDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.9200000166893005,
        ),
        (
            "mnist",
            "load_mnist_mislabeling_config",
            False,
            True,
            MislabelingDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.44353821873664856,
        ),
        (
            "mnist",
            "load_mnist_mislabeling_config",
            True,
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
            True,
            ShortcutDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.12516948580741882,
        ),
        (
            "mnist",
            "load_mnist_subclass_config",
            True,
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
    offline,
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
        offline=offline,
    )

    score = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=8,
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)


@pytest.mark.tested
@pytest.mark.parametrize(
    "test_id, config, load_from_disk, offline, bench_cls, explainer_cls, expl_kwargs, logger",
    [
        (
            "mnist",
            "load_mnist_unit_test_config",
            True,
            True,
            ClassDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            None,
        ),
        (
            "mnist",
            "load_mnist_mixed_config",
            True,
            True,
            MixedDatasets,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            None,
        ),
        (
            "mnist",
            "load_mnist_mislabeling_config",
            False,
            True,
            MislabelingDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            None,
        ),
        (
            "mnist",
            "load_mnist_shortcut_config",
            True,
            True,
            ShortcutDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            None,
        ),
        (
            "mnist",
            "load_mnist_subclass_config",
            True,
            True,
            SubclassDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            None,
        ),
    ],
)
def test_train_from_config(
    test_id,
    config,
    load_from_disk,
    offline,
    bench_cls,
    explainer_cls,
    expl_kwargs,
    logger,
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

    if logger is not None:
        logger_cfg = request.getfixturevalue(logger)
        config["log_dir"] = os.path.join(str(tmp_path), "logs")
        config["logger"] = logger_cfg
        config = OmegaConf.create(config)
        logger = BenchConfigParser.parse_logger(config)

    # create the dirs
    os.makedirs(config["ckpt_dir"], exist_ok=True)
    os.makedirs(config["metadata_dir"], exist_ok=True)

    dst_eval = bench_cls.train(
        config=config,
        logger=logger,
        load_meta_from_disk=load_from_disk,
    )

    # TODO: create push_to_hub method
    # dst_eval.model.push_to_hub(f"quanda-bench-test/{config['id']}")

    score = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=8,
    )["score"]

    assert score is not None


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, config, load_from_disk, offline, bench_cls",
    [
        (
            "mnist_topk",
            "load_mnist_unit_test_config",
            True,
            True,
            TopKCardinality,
        ),
        (
            "mnist_rand",
            "load_mnist_unit_test_config",
            True,
            True,
            ModelRandomization,
        ),
        (
            "mnist_mixed",
            "load_mnist_mixed_config",
            True,
            True,
            MixedDatasets,
        ),
        (
            "mnist_class",
            "load_mnist_unit_test_config",
            True,
            True,
            ClassDetection,
        ),
        (
            "mnist_flip",
            "load_mnist_mislabeling_config",
            False,
            True,
            MislabelingDetection,
        ),
        (
            "mnist_flip_load",
            "load_mnist_mislabeling_config",
            True,
            True,
            MislabelingDetection,
        ),
        (
            "mnist_ch_load",
            "load_mnist_shortcut_config",
            True,
            True,
            ShortcutDetection,
        ),
        (
            "mnist_ch",
            "load_mnist_shortcut_config",
            False,
            True,
            ShortcutDetection,
        ),
        (
            "mnist_sub",
            "load_mnist_subclass_config",
            True,
            True,
            SubclassDetection,
        ),
    ],
)
def test_sanity_from_config(
    test_id,
    config,
    load_from_disk,
    offline,
    bench_cls,
    tmp_path,
    request,
):
    config = request.getfixturevalue(config)

    config["cache_dir"] = str(tmp_path)
    dst_eval = bench_cls.from_config(
        config=config,
        load_meta_from_disk=load_from_disk,
    )
    sanity_results = dst_eval.sanity_check()

    assert isinstance(sanity_results["train_acc"], float)


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, config, load_from_disk, offline, logger_cfg",
    [
        (
            "mnist",
            "load_mnist_unit_test_config",
            True,
            True,
            "load_wandb_config",
        ),
        (
            "mnist",
            "load_mnist_unit_test_config",
            True,
            True,
            "load_tensorboard_config",
        ),
    ],
)
def test_logger(
    test_id,
    config,
    load_from_disk,
    offline,
    logger_cfg,
    tmp_path,
    request,
):
    config = request.getfixturevalue(config)
    logger_cfg = request.getfixturevalue(logger_cfg)

    config["log_dir"] = "./tmp"  # str(tmp_path)
    config["logger"] = logger_cfg

    config = OmegaConf.create(config)

    # to hydra object

    logger = BenchConfigParser.parse_logger(config)
    logger.log_metrics({"test": 1})
