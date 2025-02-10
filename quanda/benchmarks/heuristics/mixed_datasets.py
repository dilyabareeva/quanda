"""Mixed Datasets benchmark module."""

import logging
from typing import Callable, List, Optional, Union, Any

import lightning as L
import torch

from quanda.benchmarks.base import Benchmark
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
        self.adversarial_label: int

        self.filter_by_prediction: bool
        self.cache_dir: str
        self.checkpoints: Optional[List[str]]
        self.checkpoints_load_func: Optional[Callable[..., Any]]
        self.use_predictions: bool = False

    @classmethod
    def from_config(
        cls,
        config: dict,
        load_meta_from_disk: bool = True,
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
        device: str, optional
            Device to use for the evaluation, by default "cpu".
        """
        obj = cls()
        obj.device = device
        train_base_dataset = obj.dataset_from_cfg(
            config=config["train_dataset"],
            metadata_dir=config.get("metadata_dir"),
            load_meta_from_disk=load_meta_from_disk,
        )
        val_base_dataset = obj.dataset_from_cfg(
            config=config.get("val_dataset", None),
            metadata_dir=config.get("metadata_dir"),
            load_meta_from_disk=load_meta_from_disk,
        )
        adv_dataset = obj.dataset_from_cfg(
            config=config["adv_dataset"],
            metadata_dir=config.get("metadata_dir"),
            load_meta_from_disk=load_meta_from_disk,
        )
        adv_base_dataset, adv_val_dataset, obj.eval_dataset = (
            obj.split_dataset(
                dataset=adv_dataset,
                metadata_dir=config.get("metadata_dir"),
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

        obj.model, obj.checkpoints = obj.model_from_cfg(config=config["model"])
        obj.filter_by_prediction = config.get("filter_by_prediction", False)

        obj.checkpoints_load_func = None  # TODO: be more flexible
        return obj

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
