"""Shortcut Detection Benchmark."""

from typing import Any, Callable, List, Optional, Union

import lightning as L
import torch
from torch.utils.data import Subset

from quanda.benchmarks.base import Benchmark
from quanda.benchmarks.config_parser import BenchConfigParser
from quanda.metrics.downstream_eval.shortcut_detection import (
    ShortcutDetectionMetric,
)
from quanda.utils.common import (
    DatasetSplit,
    class_accuracy,
    load_last_checkpoint,
)
from quanda.utils.datasets.dataset_handlers import get_dataset_handler
from quanda.utils.datasets.transformed.sample import (
    SampleTransformationDataset,
)


class ShortcutDetection(Benchmark):
    # TODO: Add citation to the original paper formulating ShortcutDetection
    #  after acceptance
    """Benchmark for shortcut detection evaluation task.

    A class is selected, and a subset of its images is modified by overlaying a
    shortcut trigger. The model is then trained on this dataset and learns to
    use the shortcut as a trigger to predict the class. The objective is to
    detect this shortcut by analyzing the model's attributions.

    Note that all explanations are generated with respect to the class of the
    shortcut samples, to detect the shortcut.

    The average attributions for triggered examples from the class, clean
    examples from the class, and clean examples from other classes are
    computed.

    This metric is inspired by the Domain Mismatch Detection Test of Koh et al.
    (2017) and the Backdoor Poisoning Detection.

    References
    ----------
    1) Koh, Pang Wei, and Percy Liang. (2017). Understanding black-box
    predictions via influence functions. International conference on machine
    learning. PMLR.

    """

    name: str = "Shortcut Detection"
    eval_args = ["test_data", "test_labels", "explanations"]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initialize the benchmark object.

        This initializer is not used directly, instead,
        the `generate` or the `assemble` methods should be used.
        Alternatively, `download` can be used to load a precomputed benchmark.
        """
        super().__init__()

        self.model: Union[torch.nn.Module, L.LightningModule]

        self.train_dataset: SampleTransformationDataset
        self.eval_dataset: SampleTransformationDataset
        self.shortcut_cls: int
        self.device: str

        self.use_predictions: bool
        self.filter_by_prediction: bool
        self.filter_by_class: bool
        self.filter_indices: Optional[List[int]]
        self.checkpoints: List[str]
        self.checkpoints_load_func: Callable[..., Any]

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
        offline : bool
            If True, the model is not downloaded, default False.
        device: str, optional
            Device to use for the evaluation, by default "cpu".

        """
        obj = super().from_config(config, load_meta_from_disk, offline, device)

        if not isinstance(obj.eval_dataset, SampleTransformationDataset):
            raise ValueError(
                "Shortcut detection evaluation requires a "
                "SampleTransformationDataset as the evaluation dataset."
            )
        if not isinstance(obj.train_dataset, SampleTransformationDataset):
            raise ValueError(
                "Shortcut detection evaluation requires a "
                "SampleTransformationDataset as the training dataset."
            )

        assert isinstance(obj, ShortcutDetection), (
            "The object must be an instance of ShortcutDetection."
        )

        assert obj.train_dataset.metadata.cls_idx is not None, (
            "The training dataset must have a class index in its metadata."
        )

        obj.shortcut_cls = obj.train_dataset.metadata.cls_idx
        obj.use_predictions = config.get("use_predictions", True)
        obj.filter_by_prediction = config.get("filter_by_prediction", False)
        obj.filter_by_class = config.get("filter_by_class", False)

        cache_dir = config.get("bench_save_dir", "./tmp")
        metadata_dir = BenchConfigParser.get_metadata_dir(
            cfg=config, bench_save_dir=cache_dir
        )
        eval_ds_config = config["eval_dataset"]
        eval_indices = eval_ds_config["filter_indices"]

        if DatasetSplit.exists(metadata_dir, eval_indices["split_filename"]) and load_meta_from_disk:
            obj.filter_indices = DatasetSplit.load(
                metadata_dir,
                name=eval_indices["split_filename"]
            )[eval_indices["split_name"]]
        else:
            obj.filter_indices = None

        return obj

    @classmethod
    def train(
        cls,
        config: dict,
        logger: Optional[L.pytorch.loggers.logger.Logger] = None,
        device: str = "cpu",
        batch_size: int = 8,
    ):  # pragma: no cover
        """Train a model using the provided config and push to HF hub."""
        obj = super().train(
            config,
            logger=logger,
            device=device,
            batch_size=batch_size,
        )

        assert isinstance(obj, ShortcutDetection), "Not ShortcutDetection."

        # locate indices of eval samples that are shortcuts

        load_last_checkpoint(
            model=obj.model,
            checkpoints=obj.checkpoints,
            checkpoints_load_func=obj.checkpoints_load_func,
        )
        ds_handler = get_dataset_handler(dataset=obj.eval_dataset)
        expl_dl = ds_handler.create_dataloader(
            dataset=obj.eval_dataset,
            batch_size=batch_size,
            shuffle=False,
        )

        select_indices: List[int] = []

        for i, batch in enumerate(expl_dl):
            inputs, labels = ds_handler.process_batch(
                batch=batch,
                device=obj.device,
            )
            model_inputs = ds_handler.get_model_inputs(inputs=inputs)
            outputs = (
                obj.model(**model_inputs)
                if isinstance(model_inputs, dict)
                else obj.model(model_inputs)
            )
            pred_cls = ds_handler.get_predictions(outputs=outputs)

            select_idx = torch.tensor([True] * len(pred_cls))
            if obj.filter_by_class:
                select_idx *= pred_cls == obj.shortcut_cls
            if obj.filter_by_prediction:
                select_idx *= labels != obj.shortcut_cls
            select_indices.extend(select_idx)

        obj.filter_indices = torch.nonzero(torch.tensor(select_indices), as_tuple=False)
        obj.save_filtered_indices(config)

        return obj

    def save_filtered_indices(self, config):
        cache_dir = config.get("bench_save_dir", "./tmp")
        metadata_dir = BenchConfigParser.get_metadata_dir(
            cfg=config, bench_save_dir=cache_dir
        )
        eval_ds_config = config["eval_dataset"]
        eval_indices = eval_ds_config["filter_indices"]
        split = DatasetSplit({eval_indices["split_name"]: self.filter_indices})
        split.save(metadata_dir, eval_indices["split_filename"])

    def sanity_check(self, batch_size: int = 32) -> dict:
        """Compute accuracy on shortcut datapoints as a sanity check.

        Parameters
        ----------
        batch_size : int, optional
            Batch size to be used for the evaluation, default to 32.

        Returns
        -------
        dict
            Dictionary containing the evaluation results.

        """
        results = super().sanity_check(batch_size)

        train_dl = torch.utils.data.DataLoader(
            Subset(self.train_dataset, self.train_dataset.transform_indices),
            batch_size=batch_size,
            shuffle=False,
        )

        eval_dl = torch.utils.data.DataLoader(
            Subset(self.eval_dataset, self.eval_dataset.transform_indices),
            batch_size=batch_size,
            shuffle=False,
        )

        results["shortcut_memorization"] = class_accuracy(
            self.model, train_dl, self.device
        )
        results["eval_shortcut_classification"] = class_accuracy(
            self.model, eval_dl, single_class=self.shortcut_cls, device=self.device
        )

        return results

    def evaluate(
        self,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        batch_size: int = 8,
    ):
        """Evaluate the given data attributor.

        Parameters
        ----------
        explainer_cls : type
            Class of the explainer to be used for the evaluation.
        expl_kwargs : Optional[dict], optional
            Additional keyword arguments for the explainer, by default None.
        batch_size : int, optional
            Batch size to be used for the evaluation, default to 8.

        Returns
        -------
        Dict[str, float]
            Dictionary containing the evaluation results.

        """
        explainer = self._prepare_explainer(
            dataset=self.train_dataset,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
        )

        if self.filter_indices is not None:
            filtered_dataset = torch.utils.data.Subset(
                self.eval_dataset,
                self.filter_indices,
            )
        else:
            filtered_dataset = self.eval_dataset

        metric = ShortcutDetectionMetric(
            model=self.model,
            checkpoints=self.checkpoints,
            checkpoints_load_func=self.checkpoints_load_func,
            train_dataset=self.train_dataset,
            shortcut_indices=self.train_dataset.transform_indices,
            shortcut_cls=self.shortcut_cls,
            filter_by_prediction=self.filter_by_prediction,
            filter_by_class=self.filter_by_class,
        )
        return self._evaluate_dataset(
            eval_dataset=filtered_dataset,
            explainer=explainer,
            metric=metric,
            batch_size=batch_size,
        )
