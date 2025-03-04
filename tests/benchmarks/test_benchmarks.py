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
from quanda.benchmarks.ground_truth import LinearDatamodeling
from quanda.benchmarks.heuristics import (
    ModelRandomization,
    TopKCardinality,
    MixedDatasets,
)
from quanda.explainers.wrappers import CaptumSimilarity
from quanda.utils.functions import cosine_similarity


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, bench_id, load_from_disk, offline, bench_cls, explainer_cls, expl_kwargs, expected_score",
    [
        (
            "mnist",
            "mnist_mislabeling_detection_unit",
            False,
            False,
            MislabelingDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.44353821873664856,
        ),
        (
            "mnist",
            "mnist_top_k_cardinality_unit",
            True,
            False,
            TopKCardinality,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.638,
        ),
        (
            "mnist",
            "mnist_mixed_datasets_unit",
            True,
            False,
            MixedDatasets,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.03915480896830559,
        ),
        (
            "mnist",
            "mnist_class_detection_unit",
            True,
            False,
            ClassDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.9399999976158142,
        ),
        (
            "mnist",
            "mnist_shortcut_detection_unit",
            True,
            False,
            ShortcutDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.1348356008529663,
        ),
        (
            "mnist",
            "mnist_subclass_detection_unit",
            True,
            False,
            SubclassDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.23999999463558197,
        ),
    ],
)
def test_load(
    test_id,
    bench_id,
    load_from_disk,
    offline,
    bench_cls,
    explainer_cls,
    expl_kwargs,
    expected_score,
    tmp_path,
    request,
):
    expl_kwargs = {
        **expl_kwargs,
        "model_id": "test",
        "cache_dir": str(tmp_path),
    }

    dst_eval = bench_cls.load_pretrained(
        bench_id=bench_id,
        cache_dir=str(tmp_path),
        offline=offline,
        device="cpu",
    )

    score = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=8,
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)


@pytest.mark.benchmark
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
        (
            "mnist",
            "load_mnist_linear_datamodeling_config",
            True,
            True,
            LinearDatamodeling,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            None,
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


@pytest.mark.benchmarks
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
        (
            "mnist",
            "load_mnist_linear_datamodeling_config",
            True,
            True,
            LinearDatamodeling,
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

    config["bench_save_dir"] = str(tmp_path)

    if logger is not None:
        logger_cfg = request.getfixturevalue(logger)
        config["log_dir"] = os.path.join(str(tmp_path), "logs")
        config["logger"] = logger_cfg
        config = OmegaConf.create(config)
        logger = BenchConfigParser.parse_logger(config)

    dst_eval = bench_cls.train(
        config=config,
        logger=logger,
        # load_meta_from_disk=load_from_disk,
    )

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
