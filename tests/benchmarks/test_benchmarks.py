"""Contains tests common to all benchmarks."""

import math
import os

import pytest
import torch
import yaml
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
    MixedDatasets,
    ModelRandomization,
    TopKCardinality,
)
from quanda.benchmarks.resources import config_map
from quanda.explainers.wrappers import CaptumSimilarity
from quanda.utils.datasets.dataset_handlers import get_dataset_handler
from quanda.utils.functions import cosine_similarity


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, bench_id, load_from_disk, offline, bench_cls, explainer_cls, expl_kwargs, expected_score",
    [
        (
            "mnist",
            "mnist_mixed_datasets_unit",
            True,
            False,
            MixedDatasets,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.007874015718698502,
        ),
        (
            "mnist",
            "mnist_shortcut_detection_unit",
            True,
            False,
            ShortcutDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.051058314740657806,
        ),
        (
            "mnist",
            "mnist_mislabeling_detection_unit",
            False,
            False,
            MislabelingDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.8569128513336182,
        ),
        (
            "mnist",
            "mnist_top_k_cardinality_unit",
            True,
            False,
            TopKCardinality,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.548,
        ),
        (
            "mnist",
            "mnist_class_detection_unit",
            True,
            False,
            ClassDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.8100000023841858,
        ),
        (
            "mnist",
            "mnist_subclass_detection_unit",
            True,
            False,
            SubclassDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.12999999523162842,
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


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, bench_id, bench_cls",
    [
        (
            "mnist",
            "mnist_mixed_datasets_unit",
            MixedDatasets,
        ),
        (
            "mnist",
            "mnist_shortcut_detection_unit",
            ShortcutDetection,
        ),
        (
            "mnist",
            "mnist_mislabeling_detection_unit",
            MislabelingDetection,
        ),
        (
            "mnist",
            "mnist_top_k_cardinality_unit",
            TopKCardinality,
        ),
        (
            "mnist",
            "mnist_class_detection_unit",
            ClassDetection,
        ),
        (
            "mnist",
            "mnist_subclass_detection_unit",
            SubclassDetection,
        ),
    ],
)
def test_overall_objective(
    test_id,
    bench_id,
    bench_cls,
    tmp_path,
):
    dst_eval = bench_cls.load_pretrained(
        bench_id=bench_id,
        cache_dir=str(tmp_path),
        offline=False,
        device="cpu",
    )

    results = dst_eval.sanity_check()
    overall_objective = dst_eval.overall_objective(results)

    assert isinstance(overall_objective, float)


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, bench_id, bench_cls, dataset_attr, error_match",
    [
        (
            "mislabeling_train",
            "mnist_mislabeling_detection_unit",
            MislabelingDetection,
            "train_dataset",
            "Training dataset in Mislabeling Metric should have "
            "flipped labels",
        ),
        (
            "subclass_train",
            "mnist_subclass_detection_unit",
            SubclassDetection,
            "train_dataset",
            "The train dataset must be a LabelGroupingDataset",
        ),
        (
            "subclass_eval",
            "mnist_subclass_detection_unit",
            SubclassDetection,
            "eval_dataset",
            "The eval dataset must be a LabelGroupingDataset",
        ),
        (
            "mixed_train",
            "mnist_mixed_datasets_unit",
            MixedDatasets,
            "train_dataset",
            "Training dataset must be a ConcatDataset",
        ),
    ],
)
def test_dataset_type_validation(
    test_id,
    bench_id,
    bench_cls,
    dataset_attr,
    error_match,
    tmp_path,
):
    dst_eval = bench_cls.load_pretrained(
        bench_id=bench_id,
        cache_dir=str(tmp_path),
        offline=False,
        device="cpu",
    )

    dummy_dataset = torch.utils.data.TensorDataset(
        torch.randn(10, 1, 28, 28),
        torch.randint(0, 10, (10,)),
    )
    setattr(dst_eval, dataset_attr, dummy_dataset)

    with pytest.raises(ValueError, match=error_match):
        dst_eval.evaluate(
            explainer_cls=CaptumSimilarity,
            expl_kwargs={
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
                "model_id": "test",
                "cache_dir": str(tmp_path),
            },
            batch_size=8,
        )


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, filter_kwarg, error_match",
    [
        (
            "shortcut_pred",
            "filter_by_shortcut_pred",
            "shortcut_cls must be provided if filter_by_shortcut_pred is True",
        ),
        (
            "non_shortcut",
            "filter_by_non_shortcut",
            "shortcut_cls must be provided if filter_by_non_shortcut is True",
        ),
    ],
)
def test_filter_missing_shortcut_cls(
    test_id,
    filter_kwarg,
    error_match,
    tmp_path,
):
    dst_eval = ShortcutDetection.load_pretrained(
        bench_id="mnist_shortcut_detection_unit",
        cache_dir=str(tmp_path),
        offline=False,
        device="cpu",
    )

    with pytest.raises(ValueError, match=error_match):
        dst_eval._compute_and_save_filter_by_labels_and_prediction(
            config={},
            shortcut_cls=None,
            **{filter_kwarg: True},
        )


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, config, load_from_disk, offline, bench_cls, explainer_cls, expl_kwargs, expected_score",
    [
        (
            "mnist-mixed",
            "load_mnist_mixed_config",
            True,
            True,
            MixedDatasets,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.007874015718698502,
        ),
        (
            "mnist-class",
            "load_mnist_unit_test_config",
            True,
            True,
            ClassDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.8100000023841858,
        ),
        (
            "mnist-mislabeling",
            "load_mnist_mislabeling_config",
            False,
            True,
            MislabelingDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.45566120743751526,
        ),
        (
            "mnist-mislabeling-download",
            "load_mnist_mislabeling_config",
            True,
            True,
            MislabelingDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.45566120743751526,
        ),
        (
            "mnist-shortcut",
            "load_mnist_shortcut_config",
            True,
            True,
            ShortcutDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.22792746126651764,
        ),
        (
            "mnist-shortcut-download",
            "load_mnist_shortcut_config",
            False,
            True,
            ShortcutDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.22792746126651764,
        ),
        (
            "mnist-subclass",
            "load_mnist_subclass_config",
            True,
            True,
            SubclassDetection,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.18000000715255737,
        ),
        (
            "mnist-linear-datamodeling",
            "load_mnist_linear_datamodeling_config",
            True,
            True,
            LinearDatamodeling,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.0833333432674408,
        ),
        (
            "mnist-top-k",
            "load_mnist_unit_test_config",
            True,
            True,
            TopKCardinality,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.548,
        ),
        (
            "mnist-rand",
            "load_mnist_unit_test_config",
            True,
            True,
            ModelRandomization,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.3090123236179352,
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

    config["cache_dir"] = "bench_out"
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
@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, config, load_from_disk, offline, bench_cls, explainer_cls, expl_kwargs, logger",
    [
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
            "mnist-lds",
            "load_mnist_linear_datamodeling_config",
            True,
            True,
            LinearDatamodeling,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            None,
        ),
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

    config["log_dir"] = str(tmp_path)
    config["logger"] = logger_cfg

    config = OmegaConf.create(config)

    # to hydra object

    logger = BenchConfigParser.parse_logger(config)
    logger.log_metrics({"test": 1})


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ, reason="Skip on GitHub Actions"
)
@pytest.mark.production_bench
@pytest.mark.parametrize(
    "config_name,bench_cls",
    [
        ("mnist_shortcut_detection_unit", ShortcutDetection),
        ("mnist_mixed_datasets_unit", MixedDatasets),
        ("mnist_subclass_detection", SubclassDetection),
        ("mnist_mislabeling_detection", MislabelingDetection),
        ("mnist_class_detection", ClassDetection),
    ],
)
def test_benchmark_filters(config_name, bench_cls, tmp_path):
    """Verify filter_by_non_shortcut and filter_by_shortcut_pred in benchmark cfg work as expected on eval_dataset."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8
    bench_yaml = config_map[config_name]

    # Load the benchmark configuration
    with open(bench_yaml, "r") as f:
        cfg = yaml.safe_load(f)

    filter_by_prediction = cfg.get("filter_by_prediction", False)

    bench = bench_cls.load_pretrained(
        bench_id=config_name,
        cache_dir=str(tmp_path),
        device=device,
        offline=False,
    )
    bench.load_last_checkpoint()

    ds_handler = get_dataset_handler(dataset=bench.eval_dataset)
    expl_dl = ds_handler.create_dataloader(
        dataset=bench.eval_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    total, correct = 0, 0
    for i, batch in enumerate(expl_dl):
        inputs, labels = ds_handler.process_batch(
            batch=batch,
            device=device,
        )
        correct_idx = torch.tensor([True] * len(inputs)).to(inputs.device)

        if not filter_by_prediction:
            correct += correct_idx.sum().item()
            total += len(inputs)
            continue
        model_inputs = ds_handler.get_model_inputs(inputs=inputs)
        outputs = (
            bench.model(**model_inputs)
            if isinstance(model_inputs, dict)
            else bench.model(model_inputs)
        )
        pred_cls = ds_handler.get_predictions(outputs=outputs)
        correct_idx *= pred_cls != labels

        correct += correct_idx.sum().item()
        total += len(pred_cls)

    pct = correct / total
    assert pct == 1.0, (
        f"Expected the filtered samples to match the criteria, "
        f"but {total - correct:.0f}/{total} samples did not match "
        f"(pct={pct:.4f})."
    )
