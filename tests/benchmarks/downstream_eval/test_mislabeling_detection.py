import math

import pytest

from quanda.benchmarks.config_parser import BenchConfigParser
from quanda.benchmarks.downstream_eval import MislabelingDetection
from quanda.benchmarks.resources import sample_transforms
from quanda.explainers.wrappers import CaptumSimilarity
from quanda.utils.datasets.transformed import LabelFlippingDataset
from quanda.utils.datasets.transformed.metadata import LabelFlippingMetadata
from quanda.utils.functions import cosine_similarity


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, config, global_method, load_from_disk,explainer_cls, expl_kwargs, expected_score",
    [
        (
            "mnist",
            "load_mnist_mislabeling_config",
            "self-influence",
            False,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.44353821873664856,
        ),
        (
            "mnist",
            "load_mnist_mislabeling_config",
            "sum",
            False,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.48903006315231323,
        ),
        (
            "mnist",
            "load_mnist_mislabeling_config",
            "sum_abs",
            False,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            0.48903006315231323,
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
    dst_eval = MislabelingDetection()
    dst_eval.train_dataset = train_dataset
    dst_eval.global_method = global_method
    dst_eval.device = "cpu"
    dst_eval.mislabeling_indices = train_dataset.metadata.transform_indices
    dst_eval.model, dst_eval.checkpoints, dst_eval.checkpoints_load_func = (
        BenchConfigParser.parse_model_cfg(
            config["model"],
            config["bench_save_dir"],
            config["repo_id"], config["id"],
            True,
            "cpu",
        )
    )
    dst_eval.filter_by_prediction = config.get("filter_by_prediction", False)
    dst_eval.eval_dataset = eval_dataset

    score = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=8,
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)
