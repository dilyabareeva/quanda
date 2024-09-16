import copy
from typing import Callable, List, Optional, Union

import lightning as L
import torch
from tqdm import tqdm

from quanda.benchmarks.base import Benchmark
from quanda.metrics.downstream_eval.shortcut_detection import (
    ShortcutDetectionMetric,
)
from quanda.utils.datasets.transformed.sample import (
    SampleFnLiterals,
    SampleTransformationDataset,
    get_sample_fn,
)
from quanda.utils.training.trainer import BaseTrainer


class ShortcutDetection(Benchmark):
    """Benchmark for shortcut detection evaluation task.

    A class is selected and a portion of the images of that class are perturbed with a trigger.
    The model is then trained on this dataset, and learns a shortcut to use the trigger to predict the class.
    The goal is to detect this shortcut using the attributions of the model.

    Note that all explanations are generated with respect to the class of the poisoned samples, to detect the shortcut.

    The average attributions for triggered examples from the class, clean examples from the class,
    and clean examples from other classes are computed.

    This metric is inspired by the Domain Mismatch Detection Test of (1) and Backdoor Poisoning Detection of (2).

    Parameters
    ----------
    1) Koh, Pang Wei, and Percy Liang. "Understanding black-box predictions via influence functions."
        International conference on machine learning. PMLR, 2017.
    2) Søgaard, Anders. "Revisiting methods for finding influential examples." arXiv preprint arXiv:2111.04683 (2021).
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initializer for the benchmark object. This initializer should not be used directly.
        To instantiate the benchmark, use the `generate`, `assemble` or `download` class methods instead."""
        super().__init__()

        self.model: Union[torch.nn.Module, L.LightningModule]
        self.train_dataset: torch.utils.data.Dataset
        self.shortcut_dataset: SampleTransformationDataset
        self.dataset_transform: Optional[Callable]
        self.poisoned_indices: List[int]
        self.poisoned_cls: int
        self.poisoned_train_dl: torch.utils.data.DataLoader
        self.poisoned_val_dl: Optional[torch.utils.data.DataLoader]
        self.original_train_dl: torch.utils.data.DataLoader
        self.p: float
        self.sample_fn: Callable
        self.n_classes: int

    @classmethod
    def generate(
        cls,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: Union[str, torch.utils.data.Dataset],
        n_classes: int,
        poisoned_cls: int,
        trainer: Union[L.Trainer, BaseTrainer],
        sample_fn: Union[SampleFnLiterals, Callable],
        dataset_split: str = "train",
        dataset_transform: Optional[Callable] = None,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        p: float = 0.3,
        trainer_fit_kwargs: Optional[dict] = None,
        seed: int = 27,
        batch_size: int = 8,
        *args,
        **kwargs,
    ):
        """Generate the benchmark from scratch, with the specified parameters.

        Parameters
        ----------
        model : Union[torch.nn.Module, L.LightningModule]
            Model to be evaluated.
        train_dataset : Union[str, torch.utils.data.Dataset]
            Training dataset to be used for the benchmark. If a string is passed, it should be a HuggingFace dataset.
        n_classes : int
            Number of classes in the dataset.
        poisoned_cls : int
            The class to add triggers to.
        trainer : Union[L.Trainer, BaseTrainer]
            Trainer to be used for training the model.
        sample_fn : Union[SampleFnLiterals, Callable]
            Function to add triggers to samples of the dataset.
        dataset_split : str, optional
            Split used for HuggingFace datasets, defaults to "train"
        dataset_transform : Optional[Callable], optional
            Default transform of the dataset, defaults to None
        val_dataset : Optional[torch.utils.data.Dataset], optional
            Validation dataset to use during training, defaults to None
        p : float, optional
            The probability of poisoning with the trigger per sample, by default 0.3
        trainer_fit_kwargs : Optional[dict], optional
            Keyword arguments to supply the trainer, by default None
        seed : int, optional
            seed for reproducibility, defaults to 27
        batch_size : int, optional
            Batch size to use during training, defaults to 8

        Returns
        -------
        ShortcutDetection
            An instance of the ShortcutDetection benchmark.
        """
        obj = cls()
        obj.set_devices(model)
        obj.train_dataset = obj.process_dataset(train_dataset, dataset_split)
        obj._generate(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            p=p,
            poisoned_cls=poisoned_cls,
            sample_fn=sample_fn,
            dataset_transform=dataset_transform,
            n_classes=n_classes,
            trainer=trainer,
            trainer_fit_kwargs=trainer_fit_kwargs,
            seed=seed,
            batch_size=batch_size,
        )
        return obj

    def _generate(
        self,
        train_dataset: Union[str, torch.utils.data.Dataset],
        model: Union[torch.nn.Module, L.LightningModule],
        n_classes: int,
        poisoned_cls: int,
        sample_fn: Union[SampleFnLiterals, Callable],
        trainer: Union[L.Trainer, BaseTrainer],
        dataset_transform: Optional[Callable],
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        p: float = 0.3,
        trainer_fit_kwargs: Optional[dict] = None,
        seed: int = 27,
        batch_size: int = 8,
    ):
        """Generate the benchmark from scratch, with the specified parameters. Used internally, through the `generate` method.

        Parameters
        ----------
        model : Union[torch.nn.Module, L.LightningModule]
            Model to be evaluated.
        train_dataset : Union[str, torch.utils.data.Dataset]
            Training dataset to be used for the benchmark. If a string is passed, it should be a HuggingFace dataset.
        n_classes : int
            Number of classes in the dataset.
        poisoned_cls : int
            The class to add triggers to.
        trainer : Union[L.Trainer, BaseTrainer]
            Trainer to be used for training the model.
        sample_fn : Union[SampleFnLiterals, Callable]
            Function to add triggers to samples of the dataset.
        dataset_transform : Optional[Callable], optional
            Default transform of the dataset, defaults to None
        val_dataset : Optional[torch.utils.data.Dataset], optional
            Validation dataset to use during training, defaults to None
        p : float, optional
            The probability of poisoning with the trigger per sample, by default 0.3
        trainer_fit_kwargs : Optional[dict], optional
            Keyword arguments to supply the trainer, by default None
        seed : int, optional
            seed for reproducibility, defaults to 27
        batch_size : int, optional
            Batch size to use during training, defaults to 8

        Raises
        ------
        ValueError
            If the sample_fn is not a function or a valid literal.
        ValueError
            If the model is not a LightningModule when the trainer is a Lightning Trainer.
        ValueError
            If the model is not a torch.nn.Module when the trainer is a BaseTrainer.
        ValueError
            If the trainer is neither a Lightning Trainer nor a BaseTrainer.
        """
        if isinstance(sample_fn, str):
            sample_fn = get_sample_fn(sample_fn)
        elif not callable(sample_fn):
            raise ValueError("sample_fn should be a function or a valid literal")
        self.p = p
        self.n_classes = n_classes
        self.dataset_transform = dataset_transform
        self.poisoned_dataset = SampleTransformationDataset(
            dataset=self.train_dataset,
            p=p,
            dataset_transform=dataset_transform,
            cls_idx=poisoned_cls,
            n_classes=n_classes,
            sample_fn=sample_fn,
            seed=seed,
        )
        self.poisoned_indices = self.poisoned_dataset.transform_indices
        self.poisoned_cls = poisoned_cls
        self.sample_fn = sample_fn
        self.poisoned_train_dl = torch.utils.data.DataLoader(self.poisoned_dataset, batch_size=batch_size)
        self.original_train_dl = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size)
        if val_dataset:
            poisoned_val_dataset = SampleTransformationDataset(
                dataset=self.train_dataset,
                dataset_transform=self.dataset_transform,
                p=self.p,
                cls_idx=poisoned_cls,
                sample_fn=sample_fn,
                n_classes=self.n_classes,
            )
            self.poisoned_val_dl = torch.utils.data.DataLoader(poisoned_val_dataset, batch_size=batch_size)
        else:
            self.poisoned_val_dl = None

        self.model = copy.deepcopy(model)

        trainer_fit_kwargs = trainer_fit_kwargs or {}

        if isinstance(trainer, L.Trainer):
            if not isinstance(self.model, L.LightningModule):
                raise ValueError("Model should be a LightningModule if Trainer is a Lightning Trainer")

            trainer.fit(
                model=self.model,
                train_dataloaders=self.poisoned_train_dl,
                val_dataloaders=self.poisoned_val_dl,
                **trainer_fit_kwargs,
            )

        elif isinstance(trainer, BaseTrainer):
            if not isinstance(self.model, torch.nn.Module):
                raise ValueError("Model should be a torch.nn.Module if Trainer is a BaseTrainer")

            trainer.fit(
                model=self.model,
                train_dataloaders=self.poisoned_train_dl,
                val_dataloaders=self.poisoned_val_dl,
                **trainer_fit_kwargs,
            )

        else:
            raise ValueError("Trainer should be a Lightning Trainer or a BaseTrainer")

    @classmethod
    def download(cls, name: str, batch_size: int = 32, *args, **kwargs):
        """
        This method loads precomputed benchmark components from a file and creates an instance from the state dictionary.

        Parameters
        ----------
        name : str
            Name of the benchmark to be loaded.
        """
        bench_state = cls.download_bench_state(name)

        return cls.assemble(
            model=bench_state["model"],
            train_dataset=bench_state["train_dataset"],
            n_classes=bench_state["n_classes"],
            poisoned_indices=bench_state["poisoned_indices"],
            poisoned_cls=bench_state["poisoned_cls"],
            sample_fn=bench_state["sample_fn"],
            dataset_transform=bench_state["dataset_transform"],
            p=bench_state["p"],
            global_method=bench_state["global_method"],
            batch_size=batch_size,
        )

    @classmethod
    def assemble(
        cls,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: Union[str, torch.utils.data.Dataset],
        n_classes: int,
        sample_fn: Union[SampleFnLiterals, Callable],
        poisoned_cls: int,
        dataset_split: str = "train",
        poisoned_indices: Optional[List[int]] = None,
        dataset_transform: Optional[Callable] = None,
        p: float = 0.3,  # TODO: type specification
        batch_size: int = 8,
        *args,
        **kwargs,
    ):
        """
        Assembles the benchmark from existing components.

        Parameters
        ----------
        model : Union[torch.nn.Module, L.LightningModule]
            Model to be used for the benchmark. This model should be trained on the mislabeled dataset.
        train_dataset : Union[str, torch.utils.data.Dataset]
            Training dataset to be used for the benchmark. If a string is passed, it should be a HuggingFace dataset.
        n_classes : int
            Number of classes in the dataset.
        dataset_split : str, optional
            The dataset split, only used for HuggingFace datasets, by default "train".
        poisoned_indices : Optional[List[int]], optional
            List of indices to poison, defaults to None
        dataset_transform : Optional[Callable], optional
            Transform to be applied to the dataset, by default None
        p : float, optional
            The probability of mislabeling per sample, by default 0.3
        batch_size : int, optional
            Batch size that is used for training, by default 8
        """
        obj = cls()
        obj.model = model
        obj.train_dataset = obj.process_dataset(train_dataset, dataset_split)
        obj.p = p
        obj.dataset_transform = dataset_transform
        obj.n_classes = n_classes
        if isinstance(sample_fn, str):
            sample_fn = get_sample_fn(sample_fn)
        elif not callable(sample_fn):
            raise ValueError("sample_fn should be a function or a valid literal")

        obj.poisoned_dataset = SampleTransformationDataset(
            dataset=obj.train_dataset,
            p=p,
            cls_idx=poisoned_cls,
            dataset_transform=dataset_transform,
            sample_fn=sample_fn,
            n_classes=n_classes,
            transform_indices=poisoned_indices,
        )
        obj.poisoned_cls = poisoned_cls
        obj.poisoned_indices = obj.poisoned_dataset.transform_indices
        obj.sample_fn = sample_fn
        obj.poisoned_train_dl = torch.utils.data.DataLoader(obj.poisoned_dataset, batch_size=batch_size)
        obj.original_train_dl = torch.utils.data.DataLoader(obj.train_dataset, batch_size=batch_size)

        obj.set_devices(model)

        return obj

    def evaluate(
        self,
        expl_dataset: torch.utils.data.Dataset,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        # use_predictions: bool = False,
        batch_size: int = 8,
        *args,
        **kwargs,
    ):
        """
        Evaluate the given data attributor.

        Parameters
        ----------
        expl_dataset : torch.utils.data.Dataset
            Dataset to be used for the evaluation.
        explainer_cls : type
            Class of the explainer to be used for the evaluation.
        expl_kwargs : Optional[dict], optional
            Additional keyword arguments for the explainer, by default None
        batch_size : int, optional
            Batch size to be used for the evaluation, default to 8
        Returns
        -------
        dict
            Dictionary containing the evaluation results.
        """
        expl_kwargs = expl_kwargs or {}
        explainer = explainer_cls(model=self.model, train_dataset=self.train_dataset, **expl_kwargs)

        poisoned_expl_ds = SampleTransformationDataset(
            dataset=expl_dataset,
            dataset_transform=self.dataset_transform,
            n_classes=self.n_classes,
            sample_fn=self.sample_fn,
            p=1.0,
        )
        expl_dl = torch.utils.data.DataLoader(poisoned_expl_ds, batch_size=batch_size)
        metric = ShortcutDetectionMetric(
            model=self.model,
            train_dataset=self.poisoned_dataset,
            poisoned_indices=self.poisoned_indices,
            poisoned_cls=self.poisoned_cls,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
        )
        pbar = tqdm(expl_dl)
        n_batches = len(expl_dl)

        for i, (input, labels) in enumerate(pbar):
            pbar.set_description("Metric evaluation, batch %d/%d" % (i + 1, n_batches))

            input, labels = input.to(self.device), labels.to(self.device)
            targets = torch.ones_like(labels) * self.poisoned_cls

            explanations = explainer.explain(
                test=input,
                targets=targets,
            )
            metric.update(explanations)

        return metric.compute()
