"""Benchmark for subclass detection task."""

import copy
import logging
import os
from typing import Callable, Dict, List, Optional, Union, Any

import lightning as L
import torch
from tqdm import tqdm

from quanda.benchmarks.base import Benchmark
from quanda.benchmarks.resources import (
    load_module_from_bench_state,
    sample_transforms,
)
from quanda.benchmarks.resources.modules import bench_load_state_dict
from quanda.metrics.downstream_eval import SubclassDetectionMetric
from quanda.utils.common import ds_len, load_last_checkpoint
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

        self.group_model: Union[torch.nn.Module, L.LightningModule]
        self.base_dataset: torch.utils.data.Dataset
        self.eval_dataset: torch.utils.data.Dataset
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
        obj._set_devices(model)
        # this sets the function to the default value
        obj.checkpoints_load_func = None

        obj.base_dataset = obj._process_dataset(
            base_dataset,
            transform=dataset_transform,
            dataset_split=dataset_split,
        )
        obj.model = model
        obj.eval_dataset = eval_dataset
        obj.use_predictions = use_predictions
        obj.filter_by_prediction = filter_by_prediction

        obj._generate(
            trainer=trainer,
            cache_dir=cache_dir,
            base_dataset=base_dataset,
            dataset_transform=dataset_transform,
            val_dataset=val_dataset,
            n_classes=n_classes,
            n_groups=n_groups,
            class_to_group=class_to_group,
            trainer_fit_kwargs=trainer_fit_kwargs,
            seed=seed,
            batch_size=batch_size,
        )
        return obj

    def _generate(
        self,
        trainer: Union[L.Trainer, BaseTrainer],
        cache_dir: str,
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
        """Generate the benchmark components.

        Parameters
        ----------
        trainer : Union[L.Trainer, BaseTrainer]
            The trainer used to train the model.
        cache_dir : str
            Directory to store the generated benchmark components.
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
        self.grouped_dataset = LabelGroupingDataset(
            dataset=self.base_dataset,
            dataset_transform=dataset_transform,
            n_classes=n_classes,
            n_groups=n_groups,
            class_to_group=class_to_group,
            seed=seed,
        )

        self.class_to_group = self.grouped_dataset.class_to_group
        self.n_classes = n_classes
        self.n_groups = n_groups
        self.dataset_transform = dataset_transform

        self.grouped_train_dl = torch.utils.data.DataLoader(
            self.grouped_dataset, batch_size=batch_size
        )
        self.original_train_dl = torch.utils.data.DataLoader(
            self.base_dataset, batch_size=batch_size
        )

        if val_dataset:
            grouped_val_dataset = LabelGroupingDataset(
                dataset=self.base_dataset,
                dataset_transform=dataset_transform,
                n_classes=n_classes,
                class_to_group=self.class_to_group,
            )
            self.grouped_val_dl = torch.utils.data.DataLoader(
                grouped_val_dataset, batch_size=batch_size
            )
        else:
            self.grouped_val_dl = None

        self.group_model = copy.deepcopy(self.model).train()

        trainer_fit_kwargs = trainer_fit_kwargs or {}

        if isinstance(trainer, L.Trainer):
            if not isinstance(self.group_model, L.LightningModule):
                raise ValueError(
                    "Model should be a LightningModule if Trainer is a "
                    "Lightning Trainer"
                )

            trainer.fit(
                model=self.group_model,
                train_dataloaders=self.grouped_train_dl,
                val_dataloaders=self.grouped_val_dl,
                **trainer_fit_kwargs,
            )

        elif isinstance(trainer, BaseTrainer):
            if not isinstance(self.group_model, torch.nn.Module):
                raise ValueError(
                    "Model should be a torch.nn.Module if Trainer is a "
                    "BaseTrainer"
                )

            trainer.fit(
                model=self.group_model,
                train_dataloaders=self.grouped_train_dl,
                val_dataloaders=self.grouped_val_dl,
                **trainer_fit_kwargs,
            )

        else:
            raise ValueError(
                "Trainer should be a Lightning Trainer or a BaseTrainer"
            )

        # save check point to cache_dir
        # TODO: add model id
        torch.save(
            self.group_model.state_dict(),
            os.path.join(cache_dir, "model_subclass_detection.pth"),
        )
        self.checkpoints = [
            os.path.join(cache_dir, "model_subclass_detection.pth")
        ]  # TODO: save checkpoints
        self.group_model.to(self.device)
        self.group_model.eval()

    @classmethod
    def download(cls, name: str, cache_dir: str, device: str, *args, **kwargs):
        """Download a precomputed benchmark.

        Load precomputed benchmark components from a file and creates an
        instance from the state dictionary.

        Parameters
        ----------
        name : str
            Name of the benchmark to be loaded.
        cache_dir : str
            Directory to store the downloaded benchmark components.
        device : str
            Device to load the model on.
        args: Any
            Additional arguments.
        kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        SubclassDetection
            The benchmark instance.

        """
        obj = cls()
        bench_state = obj._get_bench_state(
            name, cache_dir, device, *args, **kwargs
        )

        checkpoint_paths = []
        for ckpt_name, ckpt in zip(
            bench_state["checkpoints"], bench_state["checkpoints_binary"]
        ):
            save_path = os.path.join(cache_dir, ckpt_name)
            torch.save(ckpt, save_path)
            checkpoint_paths.append(save_path)

        eval_dataset = obj._build_eval_dataset(
            dataset_str=bench_state["dataset_str"],
            eval_indices=bench_state["eval_test_indices"],
            transform=sample_transforms[bench_state["dataset_transform"]],
            dataset_split=bench_state["test_split_name"],
        )
        dataset_transform = sample_transforms[bench_state["dataset_transform"]]
        module = load_module_from_bench_state(bench_state, device)

        return obj.assemble(
            group_model=module,
            checkpoints=bench_state["checkpoints_binary"],
            checkpoints_load_func=bench_load_state_dict,
            base_dataset=bench_state["dataset_str"],
            n_classes=bench_state["n_classes"],
            eval_dataset=eval_dataset,
            use_predictions=bench_state["use_predictions"],
            class_to_group=bench_state["class_to_group"],
            dataset_transform=dataset_transform,
            checkpoint_paths=checkpoint_paths,
        )

    @classmethod
    def assemble(
        cls,
        group_model: Union[torch.nn.Module, L.LightningModule],
        base_dataset: Union[str, torch.utils.data.Dataset],
        n_classes: int,
        class_to_group: Dict[int, int],  # TODO: type specification
        eval_dataset: torch.utils.data.Dataset,
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        use_predictions: bool = True,
        filter_by_prediction: bool = True,
        dataset_split: str = "train",
        dataset_transform: Optional[Callable] = None,
        batch_size: int = 8,
        checkpoint_paths: Optional[List[str]] = None,
    ):
        """Assembles the benchmark from existing components.

        Parameters
        ----------
        group_model : Union[torch.nn.Module, L.LightningModule]
            The model used to generate attributions.
        base_dataset : Union[str, torch.utils.data.Dataset]
            Original dataset to use in training.
        n_classes : int
            Number of classes in `base_dataset`.
        class_to_group : Dict[int, int]
            Mapping of classes to groups.
        eval_dataset : torch.utils.data.Dataset
            Evaluation dataset to be used for the benchmark.
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

        Returns
        -------
        SubclassDetection
            The benchmark instance

        """
        obj = cls()
        obj.group_model = group_model
        obj._set_devices(group_model)
        obj.checkpoints = checkpoints
        obj.checkpoints_load_func = checkpoints_load_func
        obj.base_dataset = obj._process_dataset(
            base_dataset, transform=None, dataset_split=dataset_split
        )
        obj.class_to_group = class_to_group
        obj.dataset_transform = dataset_transform
        obj.n_classes = n_classes
        obj.eval_dataset = eval_dataset
        obj.use_predictions = use_predictions
        obj.filter_by_prediction = filter_by_prediction

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
        load_last_checkpoint(
            model=self.group_model,
            checkpoints=self.checkpoints,
            checkpoints_load_func=self.checkpoints_load_func,
        )
        self.group_model.eval()

        expl_kwargs = expl_kwargs or {}
        explainer = explainer_cls(
            model=self.group_model,
            checkpoints=self.checkpoints,
            train_dataset=self.grouped_dataset,
            checkpoints_load_func=self.checkpoints_load_func,
            **expl_kwargs,
        )

        expl_dl = torch.utils.data.DataLoader(
            self.eval_dataset, batch_size=batch_size
        )

        metric = SubclassDetectionMetric(
            model=self.group_model,
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

        pbar = tqdm(expl_dl)
        n_batches = len(expl_dl)

        for i, (inputs, labels) in enumerate(pbar):
            pbar.set_description(
                "Metric evaluation, batch %d/%d" % (i + 1, n_batches)
            )

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            grouped_labels = torch.tensor(
                [self.class_to_group[i.item()] for i in labels],
                device=labels.device,
            )
            if self.use_predictions:
                with torch.no_grad():
                    output = self.group_model(inputs)
                    targets = output.argmax(dim=-1)
            else:
                targets = grouped_labels

            explanations = explainer.explain(
                test_tensor=inputs,
                targets=targets,
            )

            metric.update(
                test_subclasses=labels,
                explanations=explanations,
                test_tensor=inputs,
                test_classes=grouped_labels,
            )

        return metric.compute()
