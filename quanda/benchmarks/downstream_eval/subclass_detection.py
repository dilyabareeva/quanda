import copy
import logging
from typing import Callable, Dict, Optional, Union

import lightning as L
import torch
from tqdm import tqdm

from quanda.benchmarks.base import Benchmark
from quanda.benchmarks.resources import (
    load_module_from_bench_state,
    sample_transforms,
)
from quanda.metrics.downstream_eval import SubclassDetectionMetric
from quanda.utils.datasets.transformed.label_grouping import (
    ClassToGroupLiterals,
    LabelGroupingDataset,
)
from quanda.utils.training.trainer import BaseTrainer

logger = logging.getLogger(__name__)


class SubclassDetection(Benchmark):
    """
    Benchmark for subclass detection tasks.

    TODO: remove USES PREDICTED LABELS, FILTERS BY CORRECT PREDICTIONS https://arxiv.org/pdf/2006.04528
    """

    name: str = "Subclass Detection"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.model: Union[torch.nn.Module, L.LightningModule]
        self.group_model: Union[torch.nn.Module, L.LightningModule]
        self.train_dataset: torch.utils.data.Dataset
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
        train_dataset: Union[str, torch.utils.data.Dataset],
        model: Union[torch.nn.Module, L.LightningModule],
        trainer: Union[L.Trainer, BaseTrainer],
        eval_dataset: torch.utils.data.Dataset,
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
        """
        This method should generate all the benchmark components and persist them in the instance.
        """

        logger.info(f"Generating {SubclassDetection.name} benchmark components based on passed arguments...")

        obj = cls()
        obj.set_devices(model)
        obj.train_dataset = obj.process_dataset(train_dataset, transform=dataset_transform, dataset_split=dataset_split)
        obj.model = model
        obj.eval_dataset = eval_dataset
        obj.use_predictions = use_predictions
        obj.filter_by_prediction = filter_by_prediction

        obj._generate(
            trainer=trainer,
            train_dataset=train_dataset,
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
        self.grouped_dataset = LabelGroupingDataset(
            dataset=self.train_dataset,
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

        self.grouped_train_dl = torch.utils.data.DataLoader(self.grouped_dataset, batch_size=batch_size)
        self.original_train_dl = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size)

        if val_dataset:
            grouped_val_dataset = LabelGroupingDataset(
                dataset=self.train_dataset,
                dataset_transform=dataset_transform,
                n_classes=n_classes,
                class_to_group=self.class_to_group,
            )
            self.grouped_val_dl = torch.utils.data.DataLoader(grouped_val_dataset, batch_size=batch_size)
        else:
            self.grouped_val_dl = None

        self.group_model = copy.deepcopy(self.model).train()

        trainer_fit_kwargs = trainer_fit_kwargs or {}

        if isinstance(trainer, L.Trainer):
            if not isinstance(self.group_model, L.LightningModule):
                raise ValueError("Model should be a LightningModule if Trainer is a Lightning Trainer")

            trainer.fit(
                model=self.group_model,
                train_dataloaders=self.grouped_train_dl,
                val_dataloaders=self.grouped_val_dl,
                **trainer_fit_kwargs,
            )

        elif isinstance(trainer, BaseTrainer):
            if not isinstance(self.group_model, torch.nn.Module):
                raise ValueError("Model should be a torch.nn.Module if Trainer is a BaseTrainer")

            trainer.fit(
                model=self.group_model,
                train_dataloaders=self.grouped_train_dl,
                val_dataloaders=self.grouped_val_dl,
                **trainer_fit_kwargs,
            )

        else:
            raise ValueError("Trainer should be a Lightning Trainer or a BaseTrainer")

    @classmethod
    def download(cls, name: str, cache_dir: str, device: str, *args, **kwargs):
        """
        This method should load the benchmark components from a file and persist them in the instance.
        """
        obj = cls()
        bench_state = obj._get_bench_state(name, cache_dir, device, *args, **kwargs)

        eval_dataset = obj.build_eval_dataset(
            dataset_str=bench_state["dataset_str"],
            eval_indices=bench_state["eval_test_indices"],
            transform=sample_transforms[bench_state["dataset_transform"]],
            dataset_split="test",
        )
        dataset_transform = sample_transforms[bench_state["dataset_transform"]]
        module = load_module_from_bench_state(bench_state, device)

        return obj.assemble(
            group_model=module,
            train_dataset=bench_state["dataset_str"],
            n_classes=bench_state["n_classes"],
            eval_dataset=eval_dataset,
            use_predictions=bench_state["use_predictions"],
            class_to_group=bench_state["class_to_group"],
            dataset_transform=dataset_transform,
        )

    @classmethod
    def assemble(
        cls,
        group_model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: Union[str, torch.utils.data.Dataset],
        n_classes: int,
        class_to_group: Dict[int, int],  # TODO: type specification
        eval_dataset: torch.utils.data.Dataset,
        use_predictions: bool = True,
        filter_by_prediction: bool = True,
        dataset_split: str = "train",
        dataset_transform: Optional[Callable] = None,
        batch_size: int = 8,
    ):
        """
        This method should assemble the benchmark components from arguments and persist them in the instance.
        """
        obj = cls()
        obj.group_model = group_model
        obj.train_dataset = obj.process_dataset(train_dataset, transform=None, dataset_split=dataset_split)
        obj.class_to_group = class_to_group
        obj.dataset_transform = dataset_transform
        obj.n_classes = n_classes
        obj.eval_dataset = eval_dataset
        obj.use_predictions = use_predictions
        obj.filter_by_prediction = filter_by_prediction

        obj.grouped_dataset = LabelGroupingDataset(
            dataset=obj.train_dataset,
            dataset_transform=dataset_transform,
            n_classes=obj.n_classes,
            class_to_group=class_to_group,
        )
        obj.grouped_train_dl = torch.utils.data.DataLoader(obj.grouped_dataset, batch_size=batch_size)
        obj.original_train_dl = torch.utils.data.DataLoader(obj.train_dataset, batch_size=batch_size)

        obj.set_devices(group_model)

        return obj

    def evaluate(
        self,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        batch_size: int = 8,
        *args,
        **kwargs,
    ):
        self.group_model.eval()

        expl_kwargs = expl_kwargs or {}
        explainer = explainer_cls(model=self.group_model, train_dataset=self.grouped_dataset, **expl_kwargs)

        expl_dl = torch.utils.data.DataLoader(self.eval_dataset, batch_size=batch_size)

        metric = SubclassDetectionMetric(
            model=self.group_model,
            train_dataset=self.grouped_dataset,
            train_subclass_labels=torch.tensor([self.grouped_dataset[s][1] for s in range(len(self.grouped_dataset))]),
            filter_by_prediction=self.filter_by_prediction,
            device=self.device,
        )

        pbar = tqdm(expl_dl)
        n_batches = len(expl_dl)

        for i, (inputs, labels) in enumerate(pbar):
            pbar.set_description("Metric evaluation, batch %d/%d" % (i + 1, n_batches))

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            grouped_labels = torch.tensor([self.class_to_group[i.item()] for i in labels], device=labels.device)
            if self.use_predictions:
                with torch.no_grad():
                    output = self.group_model(inputs)
                    targets = output.argmax(dim=-1)
            else:
                targets = grouped_labels

            explanations = explainer.explain(
                test=inputs,
                targets=targets,
            )
            # Use original labels for metric score calculation
            metric.update(grouped_labels, explanations, test_tensor=inputs, test_classes=grouped_labels)

        return metric.compute()
