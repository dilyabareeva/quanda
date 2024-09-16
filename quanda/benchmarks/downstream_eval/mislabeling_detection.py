import copy
from typing import Callable, Dict, List, Optional, Union

import lightning as L
import torch
from tqdm import tqdm

from quanda.benchmarks.base import Benchmark
from quanda.metrics.downstream_eval import MislabelingDetectionMetric
from quanda.utils.datasets.transformed.label_flipping import (
    LabelFlippingDataset,
)
from quanda.utils.training.trainer import BaseTrainer


class MislabelingDetection(Benchmark):
    """
    Benchmark for noisy label detection.
    This benchmark generates a dataset with mislabeled samples, and trains a model on it.
    Afterward, it evaluates the effectiveness of a given data attributor
    for detecting the mislabeled examples using ´quanda.metrics.downstream_eval.MislabelingDetectionMetric´.

    This is done by computing a cumulative detection curve (as in the below references), and calculating the AUC (as in (5))
    References
    ----------
    1) Koh, P. W., & Liang, P. (2017). Understanding black-box predictions via influence functions. In International
    Conference on Machine Learning (pp. 1885-1894). PMLR.

    2) Yeh, C.-K., Kim, J., Yen, I. E., Ravikumar, P., & Dhillon, I. S. (2018). Representer point selection
    for explaining deep neural networks. In Advances in Neural Information Processing Systems (Vol. 31).

    3) Pruthi, G., Liu, F., Sundararajan, M., & Kale, S. (2020). Estimating training data influence by tracing gradient
    descent. In Advances in Neural Information Processing Systems (Vol. 33, pp. 19920-19930).

    4) Picard, A. M., Vigouroux, D., Zamolodtchikov, P., Vincenot, Q., Loubes, J.-M., & Pauwels, E. (2022). Leveraging
    influence functions for dataset exploration and cleaning. In 11th European Congress on Embedded Real-Time Systems
    (ERTS 2022) (pp. 1-8). Toulouse, France.

    5) Kwon, Yongchan, et al.
        "Datainf: Efficiently estimating data influence in lora-tuned llms and diffusion models."
        arXiv preprint arXiv:2310.00902 (2023).
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initializer for the Mislabeling Detection benchmark.

        This initializer is not used directly, instead,
        the `generate` or the `assemble` methods should be used.
        Alternatively, `download` can be used to load a precomputed benchmarks.
        """
        super().__init__()

        self.model: Union[torch.nn.Module, L.LightningModule]
        self.train_dataset: torch.utils.data.Dataset
        self.poisoned_dataset: LabelFlippingDataset
        self.dataset_transform: Optional[Callable]
        self.poisoned_indices: List[int]
        self.poisoned_labels: Dict[int, int]
        self.poisoned_train_dl: torch.utils.data.DataLoader
        self.poisoned_val_dl: Optional[torch.utils.data.DataLoader]
        self.original_train_dl: torch.utils.data.DataLoader
        self.p: float
        self.global_method: Union[str, type] = "self-influence"
        self.n_classes: int

    @classmethod
    def generate(
        cls,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: Union[str, torch.utils.data.Dataset],
        n_classes: int,
        trainer: Union[L.Trainer, BaseTrainer],
        dataset_split: str = "train",
        dataset_transform: Optional[Callable] = None,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        global_method: Union[str, type] = "self-influence",
        p: float = 0.3,
        trainer_fit_kwargs: Optional[dict] = None,
        seed: int = 27,
        batch_size: int = 8,
        *args,
        **kwargs,
    ):
        """Generates the benchmark by specifying parameters.
        This module handles the dataset creation and model training on the label-poisoned dataset.
        The evaluation can then be run using the `evaluate` method.

        Parameters
        ----------
        model : Union[torch.nn.Module, L.LightningModule]
            Model to be used for the benchmark.
            Note that a new model will be trained on the label-poisoned dataset.
        train_dataset : Union[str, torch.utils.data.Dataset]
            Training dataset to be used for the benchmark. If a string is passed, it should be a HuggingFace dataset.
        n_classes : int
            Number of classes in the dataset.
        trainer : Union[L.Trainer, BaseTrainer]
            Trainer to be used for training the model. Can be a Lightning Trainer or a `BaseTrainer`.
        dataset_split : str, optional
            The dataset split, only used for HuggingFace datasets, by default "train".
        dataset_transform : Optional[Callable], optional
            Transform to be applied to the dataset, by default None
        val_dataset : Optional[torch.utils.data.Dataset], optional
            Validation dataset to be used for the benchmark, by default None
        global_method : Union[str, type], optional
            Method to generate a global ranking from local explainer.
            It can be a subclass of `quanda.explainers.aggregators.BaseAggregator` or "self-influence".
            Defaults to "self-influence".
        p : float, optional
            The probability of mislabeling per sample, by default 0.3
        trainer_fit_kwargs : Optional[dict], optional
            Additional keyword arguments for the trainer's fit method, by default None
        seed : int, optional
            Seed for reproducibility, by default 27
        batch_size : int, optional
            Batch size that is used for training, by default 8


        Returns
        -------
        MislabelingDetection
            The benchmark instance.
        """

        obj = cls()
        obj.set_devices(model)
        obj.train_dataset = obj.process_dataset(train_dataset, dataset_split)
        obj._generate(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            p=p,
            global_method=global_method,
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
        trainer: Union[L.Trainer, BaseTrainer],
        dataset_transform: Optional[Callable],
        poisoned_indices: Optional[List[int]] = None,
        poisoned_labels: Optional[Dict[int, int]] = None,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        p: float = 0.3,
        global_method: Union[str, type] = "self-influence",
        trainer_fit_kwargs: Optional[dict] = None,
        seed: int = 27,
        batch_size: int = 8,
    ):
        """Generates the benchmark from components.
        This function is internally used for generating the benchmark instance.


        Parameters
        ----------
        model : Union[torch.nn.Module, L.LightningModule]
            Model to be used for the benchmark.
            Note that a new model will be trained on the label-poisoned dataset.
        train_dataset : Union[str, torch.utils.data.Dataset]
            Training dataset to be used for the benchmark. If a string is passed, it should be a HuggingFace dataset.
        n_classes : int
            Number of classes in the dataset.
        trainer : Union[L.Trainer, BaseTrainer]
            Trainer to be used for training the model. Can be a Lightning Trainer or a `BaseTrainer`.
        dataset_transform : Optional[Callable], optional
            Transform to be applied to the dataset, by default None
        val_dataset : Optional[torch.utils.data.Dataset], optional
            Validation dataset to be used for the benchmark, by default None
        poisoned_indices : Optional[List[int]], optional
            Optional list of indices to poison, by default None
        poisoned_labels : Optional[Dict[int, int]], optional
            Optional dictionary containing indices as keys and new labels as values, by default None
        global_method : Union[str, type], optional
            Method to generate a global ranking from local explainer.
            It can be a subclass of `quanda.explainers.aggregators.BaseAggregator` or "self-influence".
            Defaults to "self-influence".
        p : float, optional
            The probability of mislabeling per sample, by default 0.3
        trainer_fit_kwargs : Optional[dict], optional
            Additional keyword arguments for the trainer's fit method, by default None
        seed : int, optional
            Seed for reproducibility, by default 27
        batch_size : int, optional
            Batch size that is used for training, by default 8

        Raises
        ------
        ValueError
            If the model is not a LightningModule and the trainer is a Lightning Trainer.
        ValueError
            If the model is not a torch.nn.Module and the trainer is a BaseTrainer.
        ValueError
            If the trainer is neither a Lightning Trainer nor a BaseTrainer.

        """
        self.p = p
        self.global_method = global_method
        self.n_classes = n_classes
        self.dataset_transform = dataset_transform
        self.poisoned_dataset = LabelFlippingDataset(
            dataset=self.train_dataset,
            p=p,
            transform_indices=poisoned_indices,
            dataset_transform=dataset_transform,
            poisoned_labels=poisoned_labels,
            n_classes=n_classes,
            seed=seed,
        )
        self.poisoned_indices = self.poisoned_dataset.transform_indices
        self.poisoned_labels = self.poisoned_dataset.poisoned_labels
        self.poisoned_train_dl = torch.utils.data.DataLoader(self.poisoned_dataset, batch_size=batch_size)
        self.original_train_dl = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size)
        if val_dataset:
            poisoned_val_dataset = LabelFlippingDataset(
                dataset=self.train_dataset, dataset_transform=self.dataset_transform, p=self.p, n_classes=self.n_classes
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
            poisoned_labels=bench_state["poisoned_labels"],
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
        dataset_split: str = "train",
        poisoned_indices: Optional[List[int]] = None,
        poisoned_labels: Optional[Dict[int, int]] = None,
        dataset_transform: Optional[Callable] = None,
        p: float = 0.3,  # TODO: type specification
        global_method: Union[str, type] = "self-influence",
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
        poisoned_labels : Optional[Dict[int, int]], optional
            Dictionary containing indices as keys and new labels as values, defaults to None
        dataset_transform : Optional[Callable], optional
            Transform to be applied to the dataset, by default None
        p : float, optional
                        The probability of mislabeling per sample, by default 0.3
        global_method : Union[str, type], optional
            Method to generate a global ranking from local explainer.
            It can be a subclass of `quanda.explainers.aggregators.BaseAggregator` or "self-influence".
            Defaults to "self-influence".
        batch_size : int, optional
            Batch size that is used for training, by default 8
        """
        obj = cls()
        obj.model = model
        obj.train_dataset = obj.process_dataset(train_dataset, dataset_split)
        obj.p = p
        obj.dataset_transform = dataset_transform
        obj.global_method = global_method
        obj.n_classes = n_classes

        obj.poisoned_dataset = LabelFlippingDataset(
            dataset=obj.train_dataset,
            p=p,
            dataset_transform=dataset_transform,
            n_classes=n_classes,
            transform_indices=poisoned_indices,
            poisoned_labels=poisoned_labels,
        )
        obj.poisoned_indices = obj.poisoned_dataset.transform_indices
        obj.poisoned_labels = obj.poisoned_dataset.poisoned_labels

        obj.poisoned_train_dl = torch.utils.data.DataLoader(obj.poisoned_dataset, batch_size=batch_size)
        obj.original_train_dl = torch.utils.data.DataLoader(obj.train_dataset, batch_size=batch_size)

        obj.set_devices(model)

        return obj

    def evaluate(
        self,
        expl_dataset: torch.utils.data.Dataset,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        use_predictions: bool = False,
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
        use_predictions : bool, optional
            Whether to use model predictions or the true test labels for the evaluation, defaults to False
        batch_size : int, optional
            Batch size to be used for the evaluation, default to 8
        Returns
        -------
        dict
            Dictionary containing the evaluation results.
        """
        expl_kwargs = expl_kwargs or {}
        explainer = explainer_cls(model=self.model, train_dataset=self.train_dataset, **expl_kwargs)

        poisoned_expl_ds = LabelFlippingDataset(
            dataset=expl_dataset, dataset_transform=self.dataset_transform, n_classes=self.n_classes, p=0.0
        )
        expl_dl = torch.utils.data.DataLoader(poisoned_expl_ds, batch_size=batch_size)
        if self.global_method != "self-influence":
            metric = MislabelingDetectionMetric.aggr_based(
                model=self.model,
                train_dataset=self.poisoned_dataset,
                poisoned_indices=self.poisoned_indices,
                aggregator_cls=self.global_method,
            )

            pbar = tqdm(expl_dl)
            n_batches = len(expl_dl)

            for i, (inputs, labels) in enumerate(pbar):
                pbar.set_description("Metric evaluation, batch %d/%d" % (i + 1, n_batches))

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if use_predictions:
                    with torch.no_grad():
                        targets = self.model(inputs).argmax(dim=-1)
                else:
                    targets = labels
                explanations = explainer.explain(test=inputs, targets=targets)
                metric.update(explanations)
        else:
            metric = MislabelingDetectionMetric.self_influence_based(
                model=self.model,
                train_dataset=self.poisoned_dataset,
                poisoned_indices=self.poisoned_indices,
                explainer_cls=explainer_cls,
                expl_kwargs=expl_kwargs,
            )

        return metric.compute()
