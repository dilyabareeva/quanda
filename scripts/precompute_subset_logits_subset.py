"""Precompute LDS counterfactual logits for a single subset index."""

import argparse
import os

import yaml

from quanda.benchmarks.ground_truth import LinearDatamodeling


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bench-save-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-eval-n", type=int, default=1000)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--inference-batch-size", type=int, default=None)
    args = parser.parse_args()

    if not os.path.isfile(args.config_path):
        raise FileNotFoundError(f"Config not found: {args.config_path}")

    with open(args.config_path) as f:
        bench_cfg = yaml.safe_load(f)
    if args.bench_save_dir:
        bench_cfg["bench_save_dir"] = args.bench_save_dir

    save_dir = LinearDatamodeling.cache_subset_logits_per_idx(
        config=bench_cfg,
        idx=args.idx,
        batch_size=args.batch_size,
        device=args.device,
        max_eval_n=args.max_eval_n,
        eval_seed=args.eval_seed,
        inference_batch_size=args.inference_batch_size,
    )
    print(f"[precompute] wrote subset {args.idx} to {save_dir}")


if __name__ == "__main__":
    main()
