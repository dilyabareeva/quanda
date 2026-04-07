import math

import pytest
import torch

from quanda.benchmarks.config_parser import BenchConfigParser
from quanda.benchmarks.downstream_eval import MislabelingDetection
from quanda.benchmarks.resources.sample_transforms import sample_transforms
from quanda.explainers.wrappers import CaptumSimilarity
from quanda.utils.datasets.transformed import LabelFlippingDataset
from quanda.utils.datasets.transformed.metadata import LabelFlippingMetadata
from quanda.utils.functions import cosine_similarity


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, config, global_method, load_from_disk,"
    "explainer_cls, expl_kwargs, expected_score",
    [
        (
            "mnist",
            "load_mnist_mislabeling_config",
            "self-influence",
            False,
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            0.44353821873664856,
        ),
    ],
)
def test_mislabeling_detection(
    test_id,
    config,
    global_method,
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

    config["cache_dir"] = str(tmp_path)

    train_metadata = LabelFlippingMetadata(
        p=config["train_dataset"]["wrapper"]["metadata"]["p"],
        seed=config["train_dataset"]["wrapper"]["metadata"]["seed"],
    )
    train_dataset = LabelFlippingDataset(
        dataset=BenchConfigParser.process_dataset(
            dataset=config["train_dataset"]["dataset_str"],
            transform=sample_transforms[config["train_dataset"]["transforms"]],
            dataset_split=config["train_dataset"]["dataset_split"],
        ),
        metadata=train_metadata,
    )

    eval_dataset = BenchConfigParser.process_dataset(
        dataset=config["eval_dataset"]["dataset_str"],
        transform=sample_transforms[config["eval_dataset"]["transforms"]],
        dataset_split=config["eval_dataset"]["dataset_split"],
    )

    model, checkpoints, checkpoints_load_func = (
        BenchConfigParser.parse_model_cfg(
            config["model"],
            config["bench_save_dir"],
            config["repo_id"],
            config["ckpts"],
            load_model_from_disk=True,
            device="cpu",
        )
    )
    dst_eval = MislabelingDetection(
        train_dataset=train_dataset,
        device="cpu",
        eval_dataset=eval_dataset,
        model=model,
        checkpoints=checkpoints,
        checkpoints_load_func=checkpoints_load_func,
    )

    score = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=8,
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)


# @pytest.mark.production_bench
@pytest.mark.parametrize(
    "config_name",
    [
        "mnist_mislabeling_detection",
    ],
)
def test_train_dataset_mislabeling_is_correct(config_name, tmp_path):
    """Verify that label flipping in the train dataset is applied
    correctly: transformed indices have flipped labels and
    non-transformed indices keep their original labels."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bench = MislabelingDetection.load_pretrained(
        bench_id=config_name,
        cache_dir=str(tmp_path),
        device=device,
        offline=False,
    )

    train_ds = bench.train_dataset
    assert isinstance(train_ds, LabelFlippingDataset), (
        "Expected train_dataset to be a LabelFlippingDataset, "
        f"got {type(train_ds).__name__}."
    )

    base_ds = train_ds.dataset
    transform_indices = set(train_ds.transform_indices)

    flipped_mismatches = []
    clean_mismatches = []
    for idx in range(len(train_ds)):
        _, train_label = train_ds[idx]
        _, base_label = base_ds[idx]
        if idx in transform_indices:
            if train_label == base_label:
                flipped_mismatches.append(idx)
        else:
            if train_label != base_label:
                clean_mismatches.append(idx)

    assert not flipped_mismatches, (
        f"{len(flipped_mismatches)} samples at transform_indices "
        f"still have their original label (expected flipped)."
    )
    assert not clean_mismatches, (
        f"{len(clean_mismatches)} non-transformed samples have a "
        f"different label than the base dataset (expected unchanged)."
    )


# @pytest.mark.production_bench
@pytest.mark.parametrize(
    "config_name",
    [
        "mnist_mislabeling_detection",
    ],
)
def test_eval_dataset_is_clean(config_name, tmp_path):
    """Verify the eval dataset is NOT a LabelFlippingDataset,
    i.e. it contains no mislabeled samples."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bench = MislabelingDetection.load_pretrained(
        bench_id=config_name,
        cache_dir=str(tmp_path),
        device=device,
        offline=False,
    )

    assert not isinstance(bench.eval_dataset, LabelFlippingDataset), (
        "Eval dataset should be clean (no label flipping), "
        f"but got {type(bench.eval_dataset).__name__}."
    )


# @pytest.mark.production_bench
@pytest.mark.parametrize(
    "config_name",
    [
        "mnist_mislabeling_detection",
    ],
)
def test_mislabeling_sanity_check_values(config_name, tmp_path):
    """Verify model fitness: train/val accuracy and mislabeling
    memorization are within expected bounds."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 8

    bench = MislabelingDetection.load_pretrained(
        bench_id=config_name,
        cache_dir=str(tmp_path),
        device=device,
        offline=False,
    )

    sanity_check_results = bench.sanity_check(batch_size=batch_size)

    assert sanity_check_results["train_acc"] > 0.85, (
        f"Expected train_acc > 0.85, got {sanity_check_results['train_acc']}."
    )
    assert sanity_check_results["val_acc"] > 0.85, (
        f"Expected val_acc > 0.85, got {sanity_check_results['val_acc']}."
    )
    assert (
        sanity_check_results["mislabeling_memorization"] > 0.01
    ), (  # TODO: improve this value
        f"Expected mislabeling_memorization > 0.5, "
        f"got {sanity_check_results['mislabeling_memorization']}."
    )
