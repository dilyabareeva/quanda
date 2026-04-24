"""Benchmark for noisy label detection."""

import logging
import os
from typing import Optional

import torch
import yaml

from quanda.benchmarks.base import (
    Benchmark,
    _hash_expl_kwargs,
    default_explanations_id,
)
from quanda.metrics.downstream_eval import MislabelingDetectionMetric
from quanda.utils.cache import ExplanationsCache
from quanda.utils.common import _subsample_dataset, class_accuracy, ds_len
from quanda.utils.datasets.transformed.label_flipping import (
    LabelFlippingDataset,
)

SELF_INFLUENCE_KEY = "self_influence"

logger = logging.getLogger(__name__)


class MislabelingDetection(Benchmark):
    """Benchmark for noisy label detection.

    This benchmark generates a dataset with mislabeled samples, and trains a
    model on it. Afterward, it evaluates the effectiveness of a given data
    attributor for detecting the mislabeled examples using
    ´quanda.metrics.downstream_eval.MislabelingDetectionMetric´.

    This is done by computing a cumulative detection curve (as described in the
    below references) and calculating the AUC following Kwon et al. (2024).

    References
    ----------
    1) Koh, P. W., & Liang, P. (2017). Understanding black-box predictions via
    influence functions. In International Conference on Machine Learning
    (pp. 1885-1894). PMLR.

    2) Yeh, C.-K., Kim, J., Yen, I. E., Ravikumar, P., & Dhillon, I. S. (2018).
    Representer point selection for explaining deep neural networks. In
    Advances in Neural Information Processing Systems (Vol. 31).

    3) Pruthi, G., Liu, F., Sundararajan, M., & Kale, S. (2020). Estimating
    training data influence by tracing gradient descent. In Advances in Neural
    Information Processing Systems (Vol. 33, pp. 19920-19930).

    4) Picard, A. M., Vigouroux, D., Zamolodtchikov, P., Vincenot, Q., Loubes,
    J.-M., & Pauwels, E. (2022). Leveraging influence functions for dataset
    exploration and cleaning. In 11th European Congress on Embedded Real-Time
    Systems (ERTS 2022) (pp. 1-8). Toulouse, France.

    5) Kwon, Y., Wu, E., Wu, K., & Zou, J. (2024). DataInf: Efficiently
    estimating data influence in LoRA-tuned LLMs and diffusion models. In The
    Twelfth International Conference on Learning Representations (pp. 1-8).

    """

    name: str = "Mislabeling Detection"
    eval_args = ["test_data", "test_labels", "explanations"]

    def sanity_check(self, batch_size: int = 32) -> dict:
        """Compute accuracy on  mislabeled datapoints as a sanity check.

        Parameters
        ----------
        batch_size : int, optional
            Batch size to be used for the evaluation, defaults to 32.

        Returns
        -------
        dict
            Dictionary containing the sanity check results.

        """
        results = super().sanity_check(batch_size)

        if not isinstance(self.train_dataset, LabelFlippingDataset):
            raise TypeError(
                "Training dataset in Mislabeling Metric should have flipped "
                "labels."
            )
        train_dl = torch.utils.data.DataLoader(
            self.train_dataset.filtered(self.train_dataset.transform_indices),
            batch_size=batch_size,
            shuffle=False,
        )

        results["mislabeling_memorization"] = class_accuracy(
            self.model, train_dl, self.device
        )

        return results

    def overall_objective(self, sanity_check_results: dict) -> float:
        """Compute overall objective score.

        Based on sanity check results, for selecting optional
        hyperparameters of the benchmark.
        Assigns extra weight to mislabeling_memorization.

        Parameters
        ----------
        sanity_check_results : dict
            Dictionary containing the results from the sanity check.

        Returns
        -------
        float
            Overall objective score computed from the sanity check results.

        """
        train_acc = sanity_check_results.get("train_acc", 0)
        val_acc = sanity_check_results.get("val_acc", 0)
        mislabeling_memorization = sanity_check_results.get(
            "mislabeling_memorization", 0
        )
        return (
            0.1 * (train_acc > 0.8)
            + 0.2 * (val_acc > 0.8)
            + 0.7 * mislabeling_memorization
        )

    def evaluate(
        self,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        batch_size: int = 8,
        max_eval_n: Optional[int] = None,
        eval_seed: int = 42,
        cache_dir: Optional[str] = None,
        use_cached_expl: bool = False,
        use_hf_expl: bool = False,
    ):
        """Evaluate the given data attributor.

        Parameters
        ----------
        explainer_cls : type
            Class of the explainer to be used for the evaluation.
        expl_kwargs : Optional[dict], optional
            Additional keyword arguments for the explainer, by default None.
        batch_size : int, optional
            Ignored for this benchmark; self-influence is driven by
            training-data, not per-eval-batch attributions.
        max_eval_n: Optional[int], optional
            Maximum number of train samples to use when computing
            self-influence (mislabeling detection has no separate eval
            loop, so this caps the train-dataset subsample). ``None``
            uses the full train dataset.
        eval_seed: int, optional
            Seed for the deterministic train-dataset subsample driven by
            ``max_eval_n``.
        cache_dir: Optional[str], optional
            Directory containing a precomputed self-influence cache,
            written by :meth:`explain`.
        use_cached_expl: bool, optional
            If True, load self-influence from ``cache_dir`` instead of
            recomputing it.
        use_hf_expl: bool, optional
            If True, download a precomputed self-influence cache from
            HF into ``cache_dir`` and load it.

        Returns
        -------
        dict
            Dictionary containing the evaluation results.

        """
        if isinstance(self.eval_dataset, LabelFlippingDataset):
            raise ValueError(
                "Evaluation dataset in Mislabeling Metric should not have "
                "flipped labels."
            )

        if not isinstance(self.train_dataset, LabelFlippingDataset):
            raise ValueError(
                "Training dataset in Mislabeling Metric should have flipped "
                "labels."
            )

        train_dataset = _subsample_dataset(
            self.train_dataset, max_n=max_eval_n, seed=eval_seed
        )
        if not isinstance(train_dataset, LabelFlippingDataset):
            raise TypeError(
                "Subsampled training dataset should still be a "
                f"LabelFlippingDataset, got {type(train_dataset).__name__}."
            )

        precomputed = self._resolve_precomputed_explanations(
            cache_dir=cache_dir,
            use_cached_expl=use_cached_expl,
            use_hf_expl=use_hf_expl,
        )
        precomputed_si: Optional[torch.Tensor] = None
        if precomputed is not None:
            precomputed_si = (
                precomputed[SELF_INFLUENCE_KEY].to(self.device).flatten()
            )
            if precomputed_si.numel() != ds_len(train_dataset):
                raise ValueError(
                    "Precomputed self-influence length "
                    f"({precomputed_si.numel()}) does not match the "
                    f"(possibly subsampled) train dataset length "
                    f"({len(train_dataset)}). Check that max_eval_n / "
                    "eval_seed match those used in `explain`."
                )

        metric = MislabelingDetectionMetric(
            model=self.model,
            checkpoints=self.checkpoints,
            checkpoints_load_func=self.checkpoints_load_func,
            train_dataset=train_dataset,
            mislabeling_indices=train_dataset.transform_indices,
            explainer_cls=explainer_cls if precomputed_si is None else None,
            expl_kwargs=expl_kwargs,
            precomputed_self_influence=precomputed_si,
        )

        return metric.compute()

    @classmethod
    def explain(
        cls,
        config: dict,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        batch_size: int = 8,
        explanations_id: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: str = "cpu",
        max_eval_n: Optional[int] = None,
        eval_seed: int = 42,
        inference_batch_size: Optional[int] = None,
    ) -> "MislabelingDetection":
        """Compute and persist self-influence scores to disk.

        Mislabeling detection is driven by training-data self-influence
        rather than per-eval-batch attributions, so the cached artifact
        is a single 1D tensor stored as ``self_influence.pt``. For
        consistency with other benchmarks, ``max_eval_n``/``eval_seed``
        here parameterize the train-dataset subsample over which
        self-influence is computed. ``inference_batch_size`` is ignored
        since there is no eval-time inference pass.
        """
        obj = cls.from_config(config, device=device)
        if explanations_id is None:
            explanations_id = default_explanations_id(
                config,
                explainer_cls,
                expl_kwargs,
                max_eval_n=max_eval_n,
                eval_seed=eval_seed,
            )

        save_dir = cache_dir or os.path.join(
            config.get("bench_save_dir", "./tmp"),
            "explanations",
            explanations_id.replace("/", "__"),
        )
        os.makedirs(save_dir, exist_ok=True)

        if not isinstance(obj, cls):
            raise TypeError(
                f"from_config returned {type(obj).__name__}, expected "
                f"{cls.__name__}."
            )
        if not isinstance(obj.train_dataset, LabelFlippingDataset):
            raise TypeError(
                "Training dataset in Mislabeling Metric should have "
                f"flipped labels, got {type(obj.train_dataset).__name__}."
            )
        train_dataset = _subsample_dataset(
            obj.train_dataset, max_n=max_eval_n, seed=eval_seed
        )
        explainer = obj._prepare_explainer(
            dataset=train_dataset,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
        )

        self_influence = explainer.self_influence(batch_size=batch_size)
        ExplanationsCache.save(
            save_dir, self_influence, num_id=SELF_INFLUENCE_KEY
        )

        safe_kwargs = {
            k: (
                v
                if isinstance(v, (str, int, float, bool, type(None)))
                else repr(v)
            )
            for k, v in (expl_kwargs or {}).items()
        }
        meta = {
            "explanations_id": explanations_id,
            "bench_id": config.get("id"),
            "bench": config.get("bench"),
            "explainer_cls": explainer_cls.__name__,
            "expl_kwargs": safe_kwargs,
            "expl_kwargs_hash": _hash_expl_kwargs(expl_kwargs),
            "batch_size": batch_size,
            "artifact": SELF_INFLUENCE_KEY,
            "max_eval_n": max_eval_n,
            "eval_seed": eval_seed,
        }
        with open(
            os.path.join(save_dir, "explanations_config.yaml"), "w"
        ) as f:
            yaml.safe_dump(meta, f)

        obj._explanations_dir = save_dir
        obj._explanations_id = explanations_id
        return obj
