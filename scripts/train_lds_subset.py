"""Train (or push) a single LDS subset model by index.

Used by parallel orchestration scripts: workers invoke this with
``--idx I`` to train subset ``I`` locally; a final serial pass invokes
it with ``--push-only`` to upload each subset checkpoint to HF Hub.
"""

import argparse
import os

import torch
import yaml

from quanda.benchmarks.ground_truth import LinearDatamodeling


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--bench-save-dir", default=None)
    parser.add_argument(
        "--push-only",
        action="store_true",
        help="Skip training and push the existing local ckpt to HF Hub.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.config_path):
        raise FileNotFoundError(args.config_path)
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    if args.device:
        config["device"] = args.device
    if args.bench_save_dir:
        config["bench_save_dir"] = args.bench_save_dir
    if args.push_only:
        LinearDatamodeling.push_subset(config=config, idx=args.idx)
        return

    device = args.device or config.get(
        "device", "cuda" if torch.cuda.is_available() else "cpu"
    )
    LinearDatamodeling.train_subset(
        config=config,
        idx=args.idx,
        device=device,
        batch_size=config["batch_size"],
        push_to_hub=False,
    )


if __name__ == "__main__":
    main()
