import copy
from typing import Callable, List, Optional, Union

import lightning as L
import torch
from tqdm import tqdm

from quanda.benchmarks.base import Benchmark
from quanda.benchmarks.resources import sample_transforms
from quanda.metrics.downstream_eval.shortcut_detection import (
    ShortcutDetectionMetric,
)
from quanda.utils.datasets.transformed.sample import (
    SampleTransformationDataset,
)
from quanda.utils.training.trainer import BaseTrainer


class ShortcutDetection(Benchmark):
    """Benchmark for shortcut detection evaluation task.

    A class is selected and a portion of the images of that class are perturbed with a trigger.
    The model is then trained on this dataset, and learns a shortcut to use the trigger to predict the class.
    The goal is to detect this shortcut using the attributions of the model.

    Note that all explanations are generated with respect to the class of the shortcut samples, to detect the shortcut.

    The average attributions for triggered examples from the class, clean examples from the class,
    and clean examples from other classes are computed.

    This metric is inspired by the Domain Mismatch Detection Test of (1) and Backdoor Poisoning Detection of (2).

    Parameters
    ----------
    1) Koh, Pang Wei, and Percy Liang. "Understanding black-box predictions via influence functions."
        International conference on machine learning. PMLR, 2017.
    2) Hammoudeh, Z., & Lowd, D. (2022). Identifying a training-set attack's target using renormalized influence
    estimation. In Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security
    (pp. 1367-1381).

    TODO: remove FILTER BY "CORRECT" PREDICTION FOR SHORTCUT implied https://arxiv.org/pdf/2201.10055
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
        self.eval_dataset: torch.utils.data.Dataset
        self.shortcut_dataset: SampleTransformationDataset
        self.dataset_transform: Optional[Callable]
        self.shortcut_indices: Union[List[int], torch.Tensor]
        self.shortcut_cls: int
        self.shortcut_train_dl: torch.utils.data.DataLoader
        self.shortcut_val_dl: Optional[torch.utils.data.DataLoader]
        self.original_train_dl: torch.utils.data.DataLoader
        self.p: float
        self.sample_fn: Callable
        self.n_classes: int
        self.use_predictions: bool
        self.filter_by_prediction: bool
        self.filter_by_class: bool

    @classmethod
    def generate(
        cls,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: Union[str, torch.utils.data.Dataset],
        n_classes: int,
        eval_dataset: torch.utils.data.Dataset,
        shortcut_cls: int,
        trainer: Union[L.Trainer, BaseTrainer],
        sample_fn: Callable,
        filter_by_prediction: bool = True,
        filter_by_class: bool = False,
        use_predictions: bool = True,
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
        eval_dataset : torch.utils.data.Dataset
            Dataset to be used for the evaluation.
        eval_dataset : torch.utils.data.Dataset
            Dataset to be used for the evaluation.
        shortcut_cls : int
            The class to add triggers to.
        trainer : Union[L.Trainer, BaseTrainer]
            Trainer to be used for training the model.
        sample_fn : Callable
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
        obj.eval_dataset = eval_dataset
        obj.filter_by_prediction = filter_by_prediction
        obj.filter_by_class = filter_by_class
        obj.use_predictions = use_predictions

        obj._generate(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            p=p,
            shortcut_cls=shortcut_cls,
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
        shortcut_cls: int,
        sample_fn: Callable,
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
        shortcut_cls : int
            The class to add triggers to.
        trainer : Union[L.Trainer, BaseTrainer]
            Trainer to be used for training the model.
        sample_fn : Callable
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
            If the model is not a LightningModule when the trainer is a Lightning Trainer.
        ValueError
            If the model is not a torch.nn.Module when the trainer is a BaseTrainer.
        ValueError
            If the trainer is neither a Lightning Trainer nor a BaseTrainer.
        """
        self.p = p
        self.n_classes = n_classes
        self.dataset_transform = dataset_transform
        self.shortcut_dataset = SampleTransformationDataset(
            dataset=self.train_dataset,
            p=p,
            dataset_transform=dataset_transform,
            cls_idx=shortcut_cls,
            n_classes=n_classes,
            sample_fn=sample_fn,
            seed=seed,
        )
        self.shortcut_indices = self.shortcut_dataset.transform_indices
        self.shortcut_cls = shortcut_cls
        self.sample_fn = sample_fn
        self.shortcut_train_dl = torch.utils.data.DataLoader(self.shortcut_dataset, batch_size=batch_size)
        self.original_train_dl = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size)
        if val_dataset:
            shortcut_val_dataset = SampleTransformationDataset(
                dataset=self.train_dataset,
                dataset_transform=self.dataset_transform,
                p=self.p,
                cls_idx=shortcut_cls,
                sample_fn=sample_fn,
                n_classes=self.n_classes,
            )
            self.shortcut_val_dl = torch.utils.data.DataLoader(shortcut_val_dataset, batch_size=batch_size)
        else:
            self.shortcut_val_dl = None

        self.model = copy.deepcopy(model).train()

        trainer_fit_kwargs = trainer_fit_kwargs or {}

        if isinstance(trainer, L.Trainer):
            if not isinstance(self.model, L.LightningModule):
                raise ValueError("Model should be a LightningModule if Trainer is a Lightning Trainer")

            trainer.fit(
                model=self.model,
                train_dataloaders=self.shortcut_train_dl,
                val_dataloaders=self.shortcut_val_dl,
                **trainer_fit_kwargs,
            )

        elif isinstance(trainer, BaseTrainer):
            if not isinstance(self.model, torch.nn.Module):
                raise ValueError("Model should be a torch.nn.Module if Trainer is a BaseTrainer")

            trainer.fit(
                model=self.model,
                train_dataloaders=self.shortcut_train_dl,
                val_dataloaders=self.shortcut_val_dl,
                **trainer_fit_kwargs,
            )

        else:
            raise ValueError("Trainer should be a Lightning Trainer or a BaseTrainer")

    @classmethod
    def download(cls, name: str, cache_dir: str, *args, **kwargs):
        """
        This method loads precomputed benchmark components from a file and creates an instance from the state dictionary.

        Parameters
        ----------
        name : str
            Name of the benchmark to be loaded.
        eval_dataset : torch.utils.data.Dataset
            Dataset to be used for the evaluation.
        """
        bench_state = super().download(name, cache_dir, *args, **kwargs)

        eval_dataset = cls.build_eval_dataset(
            dataset_str=bench_state["dataset_str"],
            eval_indices=bench_state["eval_test_indices"],
            dataset_split="test",
        )

        dataset_transform = sample_transforms[bench_state["dataset_transform"]]
        sample_fn = sample_transforms[bench_state["sample_fn"]]

        return cls.assemble(
            model=bench_state["model"],
            train_dataset=bench_state["train_dataset"],
            n_classes=bench_state["n_classes"],
            eval_dataset=eval_dataset,
            use_predictions=bench_state["use_predictions"],
            shortcut_indices=bench_state["shortcut_indices"],
            shortcut_cls=bench_state["shortcut_cls"],
            sample_fn=bench_state["sample_fn"],
            dataset_transform=dataset_transform,
        )

    @classmethod
    def assemble(
        cls,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: Union[str, torch.utils.data.Dataset],
        n_classes: int,
        eval_dataset: torch.utils.data.Dataset,
        sample_fn: Callable,
        shortcut_cls: int,
        shortcut_indices: List[int],
        filter_by_prediction: bool = True,
        filter_by_class: bool = False,
        use_predictions: bool = True,
        dataset_split: str = "train",
        dataset_transform: Optional[Callable] = None,
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
        eval_dataset : torch.utils.data.Dataset
            Dataset to be used for the evaluation.
        sample_fn : Callable
            Function to add triggers to samples of the dataset.
        dataset_split : str, optional
            The dataset split, only used for HuggingFace datasets, by default "train".
        shortcut_indices : Optional[List[int]], optional
            Binary list of indices to poison, defaults to None
        dataset_transform : Optional[Callable], optional
            Transform to be applied to the dataset, by default None
        batch_size : int, optional
            Batch size that is used for training, by default 8
        """
        obj = cls()
        obj.model = model
        obj.train_dataset = obj.process_dataset(train_dataset, dataset_split)
        obj.eval_dataset = eval_dataset
        obj.dataset_transform = dataset_transform
        obj.n_classes = n_classes
        obj.use_predictions = use_predictions
        obj.filter_by_prediction = filter_by_prediction
        obj.filter_by_class = filter_by_class
        obj.shortcut_dataset = SampleTransformationDataset(
            dataset=obj.train_dataset,
            cls_idx=shortcut_cls,
            dataset_transform=dataset_transform,
            sample_fn=sample_fn,
            n_classes=n_classes,
            transform_indices=shortcut_indices,
        )
        obj.shortcut_cls = shortcut_cls
        obj.shortcut_indices = obj.shortcut_dataset.transform_indices
        obj.sample_fn = sample_fn

        obj.set_devices(model)

        return obj

    def evaluate(
        self,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        batch_size: int = 8,
    ):
        """
        Evaluate the given data attributor.

        Parameters
        ----------
        explainer_cls : type
            Class of the explainer to be used for the evaluation.
        expl_kwargs : Optional[dict], optional
            Additional keyword arguments for the explainer, by default None
        filter_by_prediction : bool, optional
            Whether to filter the test samples to only calculate the metric on those samples, where the shortcut class
            is predicted, by default True
        filter_by_class: bool, optional
            Whether to filter the test samples to only calculate the metric on those samples, where the shortcut class
            is not assigned as the class, by default True
        batch_size : int, optional
            Batch size to be used for the evaluation, default to 8
        Returns
        -------
        dict
            Dictionary containing the evaluation results.
        """
        self.model.eval()

        self.shortcut_train_dl = torch.utils.data.DataLoader(self.shortcut_dataset, batch_size=batch_size)
        self.original_train_dl = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size)

        expl_kwargs = expl_kwargs or {}
        explainer = explainer_cls(model=self.model, train_dataset=self.train_dataset, **expl_kwargs)

        shortcut_expl_ds = SampleTransformationDataset(
            dataset=self.eval_dataset,
            dataset_transform=self.dataset_transform,
            n_classes=self.n_classes,
            sample_fn=self.sample_fn,
            p=1.0,
        )
        expl_dl = torch.utils.data.DataLoader(shortcut_expl_ds, batch_size=batch_size)
        metric = ShortcutDetectionMetric(
            model=self.model,
            train_dataset=self.shortcut_dataset,
            shortcut_indices=self.shortcut_indices,
            shortcut_cls=self.shortcut_cls,
            filter_by_prediction=self.filter_by_prediction,
            filter_by_class=self.filter_by_class,
        )
        pbar = tqdm(expl_dl)
        n_batches = len(expl_dl)

        for i, (input, labels) in enumerate(pbar):
            pbar.set_description("Metric evaluation, batch %d/%d" % (i + 1, n_batches))

            input, labels = input.to(self.device), labels.to(self.device)

            if self.use_predictions:
                with torch.no_grad():
                    output = self.model(input)
                    targets = output.argmax(dim=-1)
            else:
                targets = labels

            explanations = explainer.explain(
                test=input,
                targets=targets,
            )
            metric.update(explanations, test_tensor=input, test_labels=labels)

        return metric.compute()
