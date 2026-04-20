"""Mixed Datasets benchmark module."""

import logging
from typing import List, Optional

import torch
from torch.utils.data import Subset

from quanda.benchmarks.base import Benchmark, _resolve_ckpts
from quanda.benchmarks.config_parser import BenchConfigParser
from quanda.metrics.heuristics.mixed_datasets import MixedDatasetsMetric
from quanda.utils.common import class_accuracy, ds_len

logger = logging.getLogger(__name__)


class MixedDatasets(Benchmark):
    # TODO: remove FILTER BY "CORRECT" PREDICTION FOR BACKDOOR implied
    #  https://arxiv.org/pdf/2201.10055
    """Mixed Datasets Benchmark.

    Evaluates the performance of a given data attribution estimation method in
    identifying adversarial examples in a classification task.

    The training dataset is assumed to consist of a "clean" and "adversarial"
    subsets, whereby the number of samples in the clean dataset is
    significantly larger than the number of samples in the adversarial dataset.
    All adversarial samples are labeled with one label from the clean dataset.
    The evaluation is based on the area under the precision-recall curve
    (AUPRC), which quantifies the ranking of the influence of adversarial
    relative to clean samples. AUPRC is chosen because it provides better
    insight into performance in highly-skewed classification tasks where
    false positives are common.

    Unlike the original implementation, we only employ a single trained model,
    but we aggregate the AUPRC scores across
    multiple test samples.

    References
    ----------
    1) Hammoudeh, Z., & Lowd, D. (2022). Identifying a training-set attack's
    target using renormalized influence estimation. In Proceedings of the 2022
    ACM SIGSAC Conference on Computer and Communications Security
    (pp. 1367-1381).

    """

    name: str = "Mixed Datasets"
    eval_args: list = ["explanations", "test_data", "test_labels"]

    def __init__(
        self,
        *args,
        adversarial_label: int = 0,
        adversarial_indices: Optional[List[int]] = None,
        filter_by_prediction: bool = False,
        **kwargs,
    ):
        """Initialize the Mixed Datasets benchmark.

        Parameters
        ----------
        *args
            Positional arguments passed to the base class.
        adversarial_label : int
            The label assigned to adversarial samples.
        adversarial_indices : Optional[List[int]]
            Binary list indicating adversarial (1) vs clean (0) samples.
        filter_by_prediction: bool, optional
            Whether to filter the test samples to only calculate the metric on
            those samples, where the adversarial class
            is predicted, by default False.
        **kwargs
            Arguments passed to the base Benchmark class.

        """
        super().__init__(*args, **kwargs)
        self.adversarial_label = adversarial_label
        self.adversarial_indices = adversarial_indices or []
        self.filter_by_prediction = filter_by_prediction

    @classmethod
    def from_config(
        cls,
        config: dict,
        load_meta_from_disk: bool = True,
        offline: bool = False,
        device: str = "cpu",
        metadata_suffix: str = "",
        load_fresh: bool = False,
    ):
        """Initialize the benchmark from a dictionary.

        Parameters
        ----------
        config : dict
            Dictionary containing the configuration.
        load_meta_from_disk : str
            Loads dataset metadata from disk if True, otherwise generates
            it, default True.
        offline : bool, optional
            If True, no HTTP request is issued to the Hub, by default
            False.
        device: str, optional
            Device to use for the evaluation, by default "cpu".
        metadata_suffix: str, optional
            Suffix to add to the metadata directory name, by default "".
            User to prevent assets clashing when multiprocessing.
        load_fresh : bool, optional
            If True, force re-download of the model checkpoints from the
            Hub, overwriting the local cache. Incompatible with
            ``offline=True``. By default False.

        """
        if offline and load_fresh:
            raise ValueError(
                "offline=True and load_fresh=True are incompatible."
            )
        metadata_dir = BenchConfigParser.get_metadata_dir(
            cfg=config,
            bench_save_dir=config.get("bench_save_dir", "./tmp"),
            suffix=metadata_suffix,
        )
        splits_cfg = config.get("splits", {})
        train_base_dataset = BenchConfigParser.parse_dataset_cfg(
            ds_config=config["train_dataset"],
            metadata_dir=metadata_dir,
            load_meta_from_disk=load_meta_from_disk,
            splits_cfg=splits_cfg,
        )
        val_base_dataset = BenchConfigParser.parse_dataset_cfg(
            ds_config=config.get("val_dataset", None),
            metadata_dir=metadata_dir,
            load_meta_from_disk=load_meta_from_disk,
            splits_cfg=splits_cfg,
        )
        adv_dataset = BenchConfigParser.parse_dataset_cfg(
            ds_config=config["adv_dataset"],
            metadata_dir=metadata_dir,
            load_meta_from_disk=load_meta_from_disk,
            splits_cfg=splits_cfg,
        )
        split_datasets = BenchConfigParser.split_dataset(
            dataset=adv_dataset,
            ds_config=config["adv_dataset"],
            metadata_dir=metadata_dir,
            load_meta_from_disk=load_meta_from_disk,
            splits_cfg=splits_cfg,
        )
        adv_base_dataset = split_datasets["train"]
        adv_val_dataset = split_datasets["val"]
        eval_dataset = split_datasets["test"]

        train_dataset: torch.utils.data.Dataset = (
            torch.utils.data.ConcatDataset(
                [adv_base_dataset, train_base_dataset]
            )
        )
        datasets_to_concat = [
            d for d in [adv_val_dataset, val_base_dataset] if d is not None
        ]
        val_dataset: Optional[torch.utils.data.Dataset] = (
            None
            if not datasets_to_concat
            else torch.utils.data.ConcatDataset(datasets_to_concat)
        )

        adversarial_indices = [1] * ds_len(adv_base_dataset) + [0] * ds_len(
            train_base_dataset
        )

        model, checkpoints, checkpoints_load_func = (
            BenchConfigParser.parse_model_cfg(
                model_cfg=config["model"],
                bench_save_dir=config["bench_save_dir"],
                ckpts=_resolve_ckpts(config),
                offline=offline,
                load_fresh=load_fresh,
                device=device,
            )
        )

        return cls(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
            device=device,
            val_dataset=val_dataset,
            use_predictions=config.get("use_predictions", False),
            adversarial_label=config["adversarial_label"],
            adversarial_indices=adversarial_indices,
            filter_by_prediction=config.get("filter_by_prediction", False),
        )

    def sanity_check(self, batch_size: int = 32) -> dict:
        """Perform model sanity checks.

        Compute the accuracy on adversarial samples along with general \
        train and validation accuracy.

        Parameters
        ----------
        batch_size : int, optional
            Batch size for the evaluation, by default 32.

        Returns
        -------
        dict
            Dictionary containing the sanity check results.

        """
        results = super().sanity_check(batch_size)

        train_dl = torch.utils.data.DataLoader(
            Subset(
                self.train_dataset,
                [
                    i
                    for i in range(ds_len(self.train_dataset))
                    if self.adversarial_indices[i] != 0.0
                ],
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        eval_dl = torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        results["train_adversarial_memorization"] = class_accuracy(
            self.model, train_dl, self.device
        )
        results["eval_adversarial_memorization"] = class_accuracy(
            self.model, eval_dl, self.device
        )

        return results

    def evaluate(
        self,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        batch_size: int = 8,
        max_eval_n: Optional[int] = 1000,
        eval_seed: int = 42,
        cache_dir: Optional[str] = None,
        use_cached_expl: bool = False,
        use_hf_expl: bool = False,
    ):
        """Evaluate the benchmark using a given explanation method.

        Parameters
        ----------
        explainer_cls: type
            The explanation class inheriting from the base Explainer class to
            be used for evaluation.
        expl_kwargs: Optional[dict], optional
            Keyword arguments for the explainer, by default None.
        batch_size: int, optional
            Batch size for the evaluation, by default 8.
        max_eval_n: Optional[int], optional
            Maximum number of evaluation samples to use. If None, uses the
            entire evaluation dataset. By default 1000.
        eval_seed: int, optional
            Random seed for evaluation sampling, by default 42.
        cache_dir: Optional[str], optional
            Directory where cached explanations are stored. Required if
            `use_cached_expl` or `use_hf_expl` is True. By default None.
        use_cached_expl: bool, optional
            Whether to use cached explanations, by default False.
        use_hf_expl: bool, optional
            Whether to use Hugging Face cached explanations, by default False.
            If use_cached_expl is also True, will prioritize local cache over
            HF cache.

        Returns
        -------
        Dict[str, float]
            Dictionary containing the metric score.

        """
        if not isinstance(self.train_dataset, torch.utils.data.ConcatDataset):
            raise ValueError("Training dataset must be a ConcatDataset.")

        precomputed = self._resolve_precomputed_explanations(
            cache_dir=cache_dir,
            use_cached_expl=use_cached_expl,
            use_hf_expl=use_hf_expl,
        )
        explainer = (
            None
            if precomputed is not None
            else self._prepare_explainer(
                dataset=self.train_dataset,
                explainer_cls=explainer_cls,
                expl_kwargs=expl_kwargs,
            )
        )

        metric = MixedDatasetsMetric(
            model=self.model,
            checkpoints=self.checkpoints,
            train_dataset=self.train_dataset,
            checkpoints_load_func=self.checkpoints_load_func,
            adversarial_indices=self.adversarial_indices,
            filter_by_prediction=False,  # the dataset is already filtered
            adversarial_label=self.adversarial_label,
        )

        return self._evaluate_dataset(
            eval_dataset=self.eval_dataset,
            explainer=explainer,
            metric=metric,
            batch_size=batch_size,
            max_eval_n=max_eval_n,
            eval_seed=eval_seed,
            precomputed_explanations=precomputed,
        )

    def _compute_and_save_indices(self, config: dict, batch_size: int = 8):
        """Determine the indices of eval dataset.

        Filter by labels and predictions. By default,
        all samples are kept.

        Parameters
        ----------
        config : dict
            Benchmark configuration dictionary (needed for save path).
        batch_size : int, optional
            Batch size for the inference pass, by default 8.

        """
        super()._compute_and_save_filter_by_labels_and_prediction(
            config=config,
            batch_size=batch_size,
            filter_by_prediction=self.filter_by_prediction,
        )
