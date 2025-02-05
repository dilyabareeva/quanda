"""Benchmark for subclass detection task."""

import logging
import os
import warnings
from typing import Callable, Dict, List, Optional, Union, Any

import lightning as L
import torch

from quanda.benchmarks.base import Benchmark
from quanda.metrics.downstream_eval import SubclassDetectionMetric
from quanda.utils.common import ds_len
from quanda.utils.datasets.transformed.label_grouping import (
    ClassToGroupLiterals,
    LabelGroupingDataset,
)
from quanda.utils.training.trainer import BaseTrainer

logger = logging.getLogger(__name__)


class SubclassDetection(Benchmark):
    # TODO: remove USES PREDICTED LABELS, FILTERS BY CORRECT PREDICTIONS
    #  https://arxiv.org/pdf/2006.04528
    """Benchmark for subclass detection task.

    A model is trained on a dataset where labels are grouped into superclasses.
    The metric evaluates the performance of an attribution method in detecting
    the subclass of a test sample from its highest attributed training point.

    References
    ----------
    1) Hanawa, K., Yokoi, S., Hara, S., & Inui, K. (2021). Evaluation of
    similarity-based explanations. In International Conference on Learning
    Representations.

    """

    name: str = "Subclass Detection"
    eval_args = ["test_labels", "explanations", "test_data", "grouped_labels"]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initialize the Subclass Detection benchmark.

        This initializer is not used directly, instead,
        the `generate` or the `assemble` methods should be used.
        Alternatively, `download` can be used to load a precomputed benchmark.
        """
        super().__init__()

        self.model: Union[torch.nn.Module, L.LightningModule]
        self.base_dataset: torch.utils.data.Dataset
        self.eval_dataset: torch.utils.data.Dataset
        self.grouped_dataset: LabelGroupingDataset
        self.dataset_transform: Optional[Callable]
        self.grouped_train_dl: torch.utils.data.DataLoader
        self.grouped_val_dl: Optional[torch.utils.data.DataLoader]
        self.original_train_dl: torch.utils.data.DataLoader
        self.class_to_group: Dict[int, int]
        self.n_classes: int
        self.n_groups: int
        self.use_predictions: bool
        self.filter_by_prediction: bool

    @classmethod
    def generate(
        cls,
        base_dataset: Union[str, torch.utils.data.Dataset],
        model: Union[torch.nn.Module, L.LightningModule],
        trainer: Union[L.Trainer, BaseTrainer],
        eval_dataset: torch.utils.data.Dataset,
        cache_dir: str,
        use_predictions: bool = True,
        filter_by_prediction: bool = True,
        dataset_split: str = "train",
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        dataset_transform: Optional[Callable] = None,
        n_classes: int = 10,
        n_groups: int = 2,
        class_to_group: Union[ClassToGroupLiterals, Dict[int, int]] = "random",
        trainer_fit_kwargs: Optional[dict] = None,
        seed: int = 27,
        batch_size: int = 8,
        *args,
        **kwargs,
    ):
        """Generate the benchmark by specifying parameters.

        The evaluation can then be run using the `evaluate` method.

        Parameters
        ----------
        base_dataset : Union[str, torch.utils.data.Dataset]
            The vanilla training dataset to be used for the benchmark.
            If a string is passed, it should be a HuggingFace dataset name.
        model : Union[torch.nn.Module, L.LightningModule]
            The model used to generate attributions.
        trainer : Union[L.Trainer, BaseTrainer]
            The trainer used to train the model.
        eval_dataset : torch.utils.data.Dataset
            The evaluation dataset to be used for the benchmark.
        cache_dir : str
            Directory to store the generated benchmark components.
        use_predictions : bool, optional
            Whether to use the model's predictions for the evaluation. Original
            paper uses the model's predictions.
            Therefore, by default True.
        filter_by_prediction : bool, optional
            Whether to filter the evaluation dataset by the model's
            predictions, using only correctly classified datapoints.
            Original paper filters the dataset. Therefore, by default True.
        dataset_split : str, optional
            The dataset split, only used for HuggingFace datasets, by default
            "train".
        val_dataset : Optional[torch.utils.data.Dataset], optional
            Validation dataset to be used for the benchmark, by default None.
        dataset_transform : Optional[Callable], optional
            The original dataset transform, by default None.
        n_classes : int, optional
            Number of classes of `base_dataset`, by default 10.
        n_groups : int, optional
            Number of groups to split the classes into, by default 2.
        class_to_group : Union[ClassToGroupLiterals, Dict[int, int]], optional
            Mapping of classes to groups, as a dictionary. For random grouping,
            pass "random". By default "random".
        trainer_fit_kwargs : Optional[dict], optional
            Additional keyword arguments to be passed to the trainer's `fit`
            method, by default None.
        seed : int, optional
            Random seed for reproducibility, by default 27.
        batch_size : int, optional
            Batch size for the dataloaders, by default 8.
        args: Any
            Additional arguments.
        kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        SubclassDetection
            The benchmark instance.

        """
        logger.info(
            f"Generating {SubclassDetection.name} benchmark components based "
            f"on passed arguments..."
        )

        obj = cls()

        save_dir = os.path.join(cache_dir, "model_subclass_detection.pth")

        base_dataset = obj._process_dataset(
            base_dataset, transform=None, dataset_split=dataset_split
        )
        grouped_dataset = LabelGroupingDataset(
            dataset=base_dataset,
            dataset_transform=dataset_transform,
            n_classes=n_classes,
            n_groups=n_groups,
            class_to_group=class_to_group,
            seed=seed,
        )

        obj = obj.assemble(
            model=model,
            base_dataset=base_dataset,
            n_classes=n_classes,
            class_to_group=grouped_dataset.class_to_group,
            eval_dataset=eval_dataset,
            grouped_dataset=grouped_dataset,
            checkpoints=[save_dir],
            checkpoints_load_func=None,
            use_predictions=use_predictions,
            filter_by_prediction=filter_by_prediction,
            dataset_split=dataset_split,
            dataset_transform=dataset_transform,
            batch_size=batch_size,
        )

        if val_dataset:
            val_dataset = LabelGroupingDataset(
                dataset=val_dataset,
                dataset_transform=dataset_transform,
                n_classes=n_classes,
                class_to_group=obj.class_to_group,
            )
        obj.model = obj._train_model(
            model=obj.model,
            trainer=trainer,
            train_dataset=obj.grouped_dataset,
            val_dataset=val_dataset,
            save_dir=save_dir,
            trainer_fit_kwargs=trainer_fit_kwargs,
            batch_size=batch_size,
        )

        return obj

    @classmethod
    def assemble(
        cls,
        model: Union[torch.nn.Module, L.LightningModule],
        base_dataset: Union[str, torch.utils.data.Dataset],
        n_classes: int,
        class_to_group: Dict[int, int],  # TODO: type specification
        eval_dataset: torch.utils.data.Dataset,
        grouped_dataset: Optional[LabelGroupingDataset] = None,
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        use_predictions: bool = True,
        filter_by_prediction: bool = True,
        dataset_split: str = "train",
        dataset_transform: Optional[Callable] = None,
        batch_size: int = 8,
        checkpoint_paths: Optional[List[str]] = None,
        *args,
        **kwargs,
    ):
        """Assembles the benchmark from existing components.

        Parameters
        ----------
        model : Union[torch.nn.Module, L.LightningModule]
            The model used to generate attributions.
        base_dataset : Union[str, torch.utils.data.Dataset]
            Original dataset to use in training.
        n_classes : int
            Number of classes in `base_dataset`.
        class_to_group : Dict[int, int]
            Mapping of classes to groups.
        eval_dataset : torch.utils.data.Dataset
            Evaluation dataset to be used for the benchmark.
        grouped_dataset : Optional[LabelGroupingDataset], optional
            The grouped dataset, by default None.
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s), defaults to None.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file, takes
            (model, checkpoint path) as two arguments, by default None.
        use_predictions : bool, optional
            Whether to use the model's predictions for the evaluation, by
            default True.
        filter_by_prediction : bool, optional
            Whether to filter the evaluation dataset by the model's
            predictions, using only correctly classified datapoints, by default
            True.
        dataset_split : str, optional
            The dataset split, only used for HuggingFace datasets, by default
            "train".
        dataset_transform : Optional[Callable], optional
            The original dataset transform, by default None.
        batch_size : int, optional
            Batch size for the dataloaders, by default 8.
        checkpoint_paths : Optional[List[str]], optional
            List of paths to the checkpoints. This parameter is only used for
            downloaded benchmarks, by default None.
        args: Any
            Additional arguments.
        kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        SubclassDetection
            The benchmark instance

        """
        obj = cls()
        obj._assemble_common(
            model=model,  # TODO: we don't need model here
            eval_dataset=eval_dataset,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
            use_predictions=use_predictions,
        )
        obj.model = model

        obj.base_dataset = obj._process_dataset(
            base_dataset, transform=None, dataset_split=dataset_split
        )
        obj.class_to_group = class_to_group
        obj.dataset_transform = dataset_transform
        obj.n_classes = n_classes
        obj.filter_by_prediction = filter_by_prediction

        if grouped_dataset is not None:
            warnings.warn(
                "Using the provided grouped dataset. The class_to_group "
                "parameter will be ignored."
            )
            obj.grouped_dataset = grouped_dataset
        else:
            obj.grouped_dataset = LabelGroupingDataset(
                dataset=obj.base_dataset,
                dataset_transform=dataset_transform,
                n_classes=obj.n_classes,
                class_to_group=class_to_group,
            )
        obj.grouped_train_dl = torch.utils.data.DataLoader(
            obj.grouped_dataset, batch_size=batch_size
        )
        obj.original_train_dl = torch.utils.data.DataLoader(
            obj.base_dataset, batch_size=batch_size
        )

        obj._checkpoint_paths = checkpoint_paths

        return obj

    def evaluate(
        self,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        batch_size: int = 8,
        *args,
        **kwargs,
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
        args: Any
            Additional arguments.
        kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        Dict[str, float]
            Dictionary containing the metric score.

        """
        explainer = self._prepare_explainer(
            dataset=self.grouped_dataset,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
        )

        metric = SubclassDetectionMetric(
            model=self.model,
            checkpoints=self.checkpoints,
            train_dataset=self.grouped_dataset,
            checkpoints_load_func=self.checkpoints_load_func,
            train_subclass_labels=torch.tensor(
                [
                    self.base_dataset[s][1]
                    for s in range(ds_len(self.base_dataset))
                ]
            ),
            filter_by_prediction=self.filter_by_prediction,
        )

        return self._evaluate_dataset(
            eval_dataset=self.eval_dataset,
            explainer=explainer,
            metric=metric,
            batch_size=batch_size,
        )
