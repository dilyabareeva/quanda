import os
import sys

sys.path.append(os.getcwd())

import torch

from argparse import ArgumentParser

from quanda.benchmarks.heuristics import (
    MixedDatasets,
    TopKCardinality,
    ModelRandomization,
)
from quanda.benchmarks.ground_truth import LinearDatamodeling
from quanda.benchmarks.downstream_eval import (
    MislabelingDetection,
    ClassDetection,
    ShortcutDetection,
    SubclassDetection,
)


def main(benchmarks_path, cache_dir):
    torch.set_float32_matmul_precision("medium")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = os.listdir(benchmarks_path)
    for f in files:
        split_array = f.replace(".pth", "").split("_")
        benchmark_name = "_".join(split_array[1:])
        if not f.endswith(".pth"):
            break
        bench_state = torch.load(
            os.path.join(benchmarks_path, f), map_location=device
        )
        benchmark_cls_dict = {
            "class_detection": ClassDetection,
            "subclass_detection": SubclassDetection,
            "shortcut_detection": ShortcutDetection,
            "mislabeling_detection": MislabelingDetection,
            "mixed_datasets": MixedDatasets,
            "model_randomization": ModelRandomization,
            "topk_cardinality": TopKCardinality,
            "linear_datamodeling": LinearDatamodeling,
        }
        benchmark = benchmark_cls_dict[benchmark_name]()
        benchmark = benchmark._parse_bench_state(
            bench_state, cache_dir, device=device
        )
        benchmark.evaluate()


if __name__ == "__main__":
    parser = ArgumentParser()

    # Define argument for method with choices
    parser.add_argument(
        "--benchmarks_path",
        required=True,
        type=str,
        help="Path to the benchmarks folder",
    )
    parser.add_argument(
        "--ignore",
        nargs="+",
        default=[],
        type=int,
        help="Ignore specific benchmarks",
    )
    parser.add_argument(
        "--cache_dir",
        required=False,
        type=str,
        default=None,
        help="Directory to use as benchmark download cache",
    )
    args = parser.parse_args()
    main(benchmarks_path=args.benchmarks_path, cache_dir=args.cache_dir)
