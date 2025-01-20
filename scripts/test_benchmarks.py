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
        if not f.endswith(".pth"):
            break
        bench_state = torch.load(
            os.path.join(benchmarks_path, f), map_location=device
        )
        if "class_detection" in f:
            benchmark = ClassDetection()
        elif "subclass" in f:
            benchmark = SubclassDetection()
        elif "shortcut" in f:
            benchmark = ShortcutDetection()
        elif "mislabeling" in f:
            benchmark = MislabelingDetection()
        elif "mixed" in f:
            benchmark = MixedDatasets()
        elif "topk" in f:
            benchmark = TopKCardinality()
        elif "model_randomization" in f:
            benchmark = ModelRandomization()
        elif "linear_datamodeling" in f:
            benchmark = LinearDatamodeling()
        benchmark._parse_bench_state(bench_state, cache_dir, device=device)


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
