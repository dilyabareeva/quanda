"""Mixed Datasets benchmark module."""

import logging
from typing import Callable, List, Optional, Union, Any

import lightning as L
import torch
from torch.utils.data import Subset

from quanda.benchmarks.config_parser import BenchConfigParser
from quanda.benchmarks.base import Benchmark
from quanda.utils.common import class_accuracy
from quanda.metrics.heuristics.mixed_datasets import MixedDatasetsMetric

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
        **kwargs,
    ):
        """Initialize the Mixed Datasets benchmark.

        This initializer is not used directly, instead,
        the `generate` or the `assemble` methods should be used.
        Alternatively, `download` can be used to load a precomputed benchmark.
        """
        super().__init__()

        self.model: Union[torch.nn.Module, L.LightningModule]

        self.train_dataset: torch.utils.data.ConcatDataset
        self.eval_dataset: torch.utils.data.Dataset
        self.val_dataset: Optional[torch.utils.data.Dataset] = None
        self.adversarial_label: int
        self.adversarial_indices: List[int]

        self.filter_by_prediction: bool
        self.cache_dir: str
        self.checkpoints: List[str]
        self.checkpoints_load_func: Callable[..., Any]
        self.use_predictions: bool = False

    @classmethod
    def from_config(
        cls,
        config: dict,
        load_meta_from_disk: bool = True,
        offline: bool = False,
        device: str = "cpu",
    ):
        """Initialize the benchmark from a dictionary.

        Parameters
        ----------
        config : dict
            Dictionary containing the configuration.
        load_meta_from_disk : str
            Loads dataset metadata from disk if True, otherwise generates it,
            default True.
        offline : bool, optional
            Whether to load the model in offline mode, by default False.
        device: str, optional
            Device to use for the evaluation, by default "cpu".

        """
        obj = cls()
        obj.device = device

        metadata_dir = BenchConfigParser.load_metadata(
            cfg=config,
            bench_save_dir=config.get("bench_save_dir", "./tmp"),
            load_meta_from_disk=load_meta_from_disk,
        )
        train_base_dataset = BenchConfigParser.parse_dataset_cfg(
            ds_config=config["train_dataset"],
            metadata_dir=metadata_dir,
            bench_save_dir=config.get("bench_save_dir", "./tmp"),
            load_meta_from_disk=load_meta_from_disk,
        )
        val_base_dataset = BenchConfigParser.parse_dataset_cfg(
            ds_config=config.get("val_dataset", None),
            metadata_dir=metadata_dir,
            bench_save_dir=config.get("bench_save_dir", "./tmp"),
            load_meta_from_disk=load_meta_from_disk,
        )
        adv_dataset = BenchConfigParser.parse_dataset_cfg(
            ds_config=config["adv_dataset"],
            metadata_dir=metadata_dir,
            bench_save_dir=config.get("bench_save_dir", "./tmp"),
            load_meta_from_disk=load_meta_from_disk,
        )
        adv_base_dataset, adv_val_dataset, obj.eval_dataset = (
            BenchConfigParser.split_dataset(
                dataset=adv_dataset,
                metadata_dir=metadata_dir,
                split_filename=config["adv_dataset"]["split_filename"],
                load_meta_from_disk=load_meta_from_disk,
            )
        )
        obj.train_dataset = torch.utils.data.ConcatDataset(
            [adv_base_dataset, train_base_dataset]
        )
        datasets_to_concat = [
            d for d in [adv_val_dataset, val_base_dataset] if d is not None
        ]
        obj.val_dataset = (
            None
            if not datasets_to_concat
            else torch.utils.data.ConcatDataset(datasets_to_concat)
        )
        obj.adversarial_label = config["adversarial_label"]
        obj.adversarial_indices = [1] * len(adv_base_dataset) + [0] * len(
            train_base_dataset
        )

        obj.model, obj.checkpoints, obj.checkpoints_load_func = (
            BenchConfigParser.parse_model_cfg(
                model_cfg=config["model"],
                bench_save_dir=config["bench_save_dir"],
                repo_id=config["repo_id"],
                cfg_id=config["id"],
                offline=offline,
                device=device,
            )
        )
        obj.filter_by_prediction = config.get("filter_by_prediction", False)

        return obj

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
                    for i in range(len(self.train_dataset))
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

        results["adversarial_memorization"] = class_accuracy(
            self.model, train_dl, self.device
        )
        results["eval_adversarial_classification"] = class_accuracy(
            self.model, eval_dl, self.device
        )

        return results

    def evaluate(
        self,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        batch_size: int = 8,
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

        Returns
        -------
        Dict[str, float]
            Dictionary containing the metric score.

        """
        if not isinstance(self.train_dataset, torch.utils.data.ConcatDataset):
            raise ValueError("Training dataset must be a ConcatDataset.")

        explainer = self._prepare_explainer(
            dataset=self.train_dataset,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
        )

        metric = MixedDatasetsMetric(
            model=self.model,
            checkpoints=self.checkpoints,
            train_dataset=self.train_dataset,
            checkpoints_load_func=self.checkpoints_load_func,
            adversarial_indices=self.adversarial_indices,
            filter_by_prediction=self.filter_by_prediction,
            adversarial_label=self.adversarial_label,
        )

        return self._evaluate_dataset(
            eval_dataset=self.eval_dataset,
            explainer=explainer,
            metric=metric,
            batch_size=batch_size,
        )
