import os
from typing import Callable, Dict, Optional, Union

import torch

from src.explainers.functional import ExplainFunc
from src.explainers.wrappers.captum_influence import captum_similarity_explain
from src.metrics.localization.identical_class import IdenticalClass
from src.toy_benchmarks.base import ToyBenchmark
from src.utils.datasets.group_label_dataset import (
    ClassToGroupLiterals,
    GroupLabelDataset,
)
from src.utils.training.trainer import Trainer


class SubclassDetection(ToyBenchmark):

    def __init__(
        self,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        super().__init__(device=device)
        self.trainer = None
        self.model = None
        self.train_dataset = None
        self.dataset_transform = None
        self.grouped_train_dl = None
        self.original_train_dl = None
        self.bench_state = None

    @classmethod
    def generate(
        cls,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        optimizer: Callable,
        lr: float,
        criterion: torch.nn.modules.loss._Loss,
        scheduler: Optional[Callable] = None,
        optimizer_kwargs: Optional[dict] = None,
        scheduler_kwargs: Optional[dict] = None,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        n_classes: int = 10,
        n_groups: int = 2,
        class_to_group: Union[ClassToGroupLiterals, Dict[int, int]] = "random",
        trainer_kwargs: Optional[dict] = None,
        seed: Optional[int] = 27,
        batch_size: int = 8,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        """
        This method should generate all the benchmark components and persist them in the instance.
        """

        obj = cls(device=device)

        obj.trainer = Trainer.from_arguments(
            model=model,
            optimizer=optimizer,
            lr=lr,
            scheduler=scheduler,
            criterion=criterion,
            optimizer_kwargs=optimizer_kwargs,
            scheduler_kwargs=scheduler_kwargs,
        )

        if obj.trainer is None:
            raise ValueError(
                "Trainer not initialized. Please initialize trainer using init_trainer_from_lightning_module or "
                "init_trainer_from_train_arguments"
            )

        obj.train_dataset = train_dataset
        grouped_dataset = GroupLabelDataset(
            dataset=train_dataset,
            n_classes=n_classes,
            n_groups=n_groups,
            class_to_group=class_to_group,
            seed=seed,
        )
        obj.class_to_group = grouped_dataset.class_to_group
        obj.grouped_train_dl = torch.utils.data.DataLoader(grouped_dataset, batch_size=batch_size)
        obj.original_train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        if val_dataset:
            grouped_val_dataset = GroupLabelDataset(
                dataset=train_dataset,
                class_to_group=obj.class_to_group,
            )
            obj.val_loader = torch.utils.data.DataLoader(grouped_val_dataset, batch_size=batch_size)
        else:
            obj.val_loader = None

        obj.model = obj.trainer.fit(
            train_loader=obj.grouped_train_dl,
            val_loader=obj.val_loader,
            trainer_kwargs=trainer_kwargs,
        )

        obj.bench_state = {
            "model": obj.model,
            "train_dataset": obj.train_dataset,  # ok this probably won't work, but that's the idea
            "class_to_group": class_to_group,
        }

        return obj

    @classmethod
    def load(cls, path: str, device: str = "cpu", batch_size: int = 8, *args, **kwargs):
        """
        This method should load the benchmark components from a file and persist them in the instance.
        """
        obj = cls(device=device)
        obj.bench_state = torch.load(path)
        obj.model = obj.bench_state["model"]
        obj.train_dataset = obj.bench_state["train_dataset"]
        obj.class_to_group = obj.bench_state["class_to_group"]

        grouped_dataset = GroupLabelDataset(
            dataset=obj.train_dataset,
            class_to_group=obj.class_to_group,
        )
        obj.grouped_train_dl = torch.utils.data.DataLoader(grouped_dataset, batch_size=batch_size)
        obj.original_train_dl = torch.utils.data.DataLoader(obj.train_dataset, batch_size=batch_size)

    @classmethod
    def assemble(
        cls,
        model: torch.nn.Module,
        train_dataset: torch.utils.data.Dataset,
        class_to_group: Dict[int, int],  # TODO: type specification
        batch_size: int = 8,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        """
        This method should assemble the benchmark components from arguments and persist them in the instance.
        """
        obj = cls(device=device)
        obj.model = model
        obj.train_dataset = train_dataset
        obj.class_to_group = class_to_group

        grouped_dataset = GroupLabelDataset(
            dataset=train_dataset,
            class_to_group=class_to_group,
        )
        obj.grouped_train_dl = torch.utils.data.DataLoader(grouped_dataset, batch_size=batch_size)
        obj.original_train_dl = torch.utils.data.DataLoader(obj.train_dataset, batch_size=batch_size)

    def save(self, path: str, *args, **kwargs):
        """
        This method should save the benchmark components to a file/folder.
        """
        torch.save(self.bench_state, path)

    """
    @classmethod
    def generate_from_pl(cls, model: torch.nn.Module, pl_module: L.LightningModule, device: str = "cpu", *args, **kwargs):
        obj = cls.__new__(cls)
        super(SubclassDetection, obj).__init__()
        obj.device = device
        obj.trainer = Trainer.from_lightning_module(model, pl_module)
        return obj

    @classmethod
    def generate_from_trainer(cls, trainer: BaseTrainer, device: str = "cpu", *args, **kwargs):
        obj = cls.__new__(cls)
        super(SubclassDetection, obj).__init__()
        if isinstance(trainer, BaseTrainer):
            obj.trainer = trainer
            obj.device = device
        else:
            raise ValueError("trainer must be an instance of BaseTrainer")
        return obj
    """

    def evaluate(
        self,
        expl_dataset: torch.utils.data.Dataset,
        explain_fn: ExplainFunc = captum_similarity_explain,
        explain_kwargs: Optional[dict] = None,
        cache_dir: str = "./cache",
        model_id: str = "default_model_id",
        batch_size: int = 8,
        device: str = "cpu",
        *args,
        **kwargs,
    ):
        grouped_expl_ds = GroupLabelDataset(
            dataset=expl_dataset,
            class_to_group=self.class_to_group,
        )  # TODO: change to class_to_group
        expl_dl = torch.utils.data.DataLoader(grouped_expl_ds, batch_size=batch_size)

        metric = IdenticalClass(model=self.model, train_dataset=self.train_dataset, device="cpu")

        for input, labels in expl_dl:
            input, labels = input.to(device), labels.to(device)
            explanations = explain_fn(
                model=self.model,
                model_id=model_id,
                cache_dir=os.path.join(cache_dir),
                train_dataset=self.train_dataset,
                test_tensor=input,
                device=device,
                **explain_kwargs,
            )
            metric.update(labels, explanations)

        return metric.compute()
