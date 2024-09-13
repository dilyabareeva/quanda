import copy
from typing import Callable, List, Optional, Union

import lightning as L
import torch
from tqdm import tqdm

from quanda.benchmarks.base import Benchmark
from quanda.metrics.heuristics.mixed_datasets import MixedDatasetsMetric
from quanda.utils.common import ds_len
from quanda.utils.datasets import (
    SingleClassImageDataset,
)
from quanda.utils.training.trainer import BaseTrainer


class MixedDatasets(Benchmark):
    """
    Benchmark that measures the performance of a given influence estimation method in separating dataset sources.

    Evaluates the performance of a given influence estimation method in identifying adversarial examples in a
    classification task.

    The training dataset is assumed to consist of a "clean" and "adversarial" subsets, whereby the number of samples
    in the clean dataset is significantly larger than the number of samples in the adversarial dataset. All adversarial
    samples are labeled with one lable from the clean dataset. The evaluation is based on the area under the
    precision-recall curve (AUPRC), which quantifies the ranking of the influence of adversarial relative to clean
    samples. AUPRC is chosen because it provides better insight into
    performance in highly-skewed classification tasks where false positives are common.

    Unlike the original implementation, we only employ a single trained model, but we aggregate the AUPRC scores across
    multiple test samples.

    References
    ----------
    1) Hammoudeh, Z., & Lowd, D. (2022). Identifying a training-set attack's target using renormalized influence
    estimation. In Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security
    (pp. 1367-1381).
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        Initializer for the Mixed Dataset metric.

        Parameters
        ----------
        args: Any
            Additional positional arguments.
        kwargs: Any
            Additional keyword arguments.

        """
        super().__init__()

        self.model: Union[torch.nn.Module, L.LightningModule]
        self.mixed_dataset: torch.utils.data.ConcatDataset
        self.adversarial_indices: List[int]
        self.adversarial_label: int
        self.adversarial_transform: Optional[Callable]

    @classmethod
    def generate(
        cls,
        model: Union[torch.nn.Module, L.LightningModule],
        clean_dataset: Union[str, torch.utils.data.Dataset],
        adversarial_dir: str,
        adversarial_label: int,
        trainer: Union[L.Trainer, BaseTrainer],
        dataset_split: str = "train",
        adversarial_transform: Optional[Callable] = None,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        trainer_fit_kwargs: Optional[dict] = None,
        batch_size: int = 8,
        *args,
        **kwargs,
    ):
        """Generates the benchmark by specifying parameters.
         This module handles the dataset creation and model training on the mixed dataset.
         The evaluation can then be run using the `evaluate` method.

        Parameters
        ----------
        model: Union[torch.nn.Module, L.LightningModule]
            Model to be used for the benchmark.
        clean_dataset: Union[str, torch.utils.data.Dataset]
            Clean dataset to be used for the benchmark. If a string is passed, it should be a HuggingFace dataset.
        adversarial_dir: str
            Directory containing the adversarial dataset. Typically consists of the same class of objects (e.g. images
            of the same class).
        adversarial_label: int
            The label to be used for the adversarial dataset.
        trainer: Union[L.Trainer, BaseTrainer]
            Trainer to be used for training the model. Can be a Lightning Trainer or a `BaseTrainer`.
        dataset_split: str, optional
            The dataset split, only used for HuggingFace datasets, by default "train".
        adversarial_transform : Optional[Callable], optional
             Transform to be applied to the adversarial dataset, by default None
        val_dataset: Optional[torch.utils.data.Dataset], optional
            Validation dataset to be used for the benchmark, by default None
        trainer_fit_kwargs: Optional[dict], optional
            Additional keyword arguments for the trainer's fit method, by default None
        batch_size: int, optional
            Batch size that is used for training, by default 8
        args: Any
            Additional positional arguments.
        kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        MixedDatasets
            The benchmark instance.
        """
        obj = cls()
        obj.set_devices(model)
        obj.clean_dataset = clean_dataset

        obj._generate(
            model=model,
            clean_dataset=clean_dataset,
            adversarial_dir=adversarial_dir,
            adversarial_label=adversarial_label,
            trainer=trainer,
            adversarial_transform=adversarial_transform,
            val_dataset=val_dataset,
            trainer_fit_kwargs=trainer_fit_kwargs,
            batch_size=batch_size,
        )
        return obj

    def _generate(
        self,
        model: Union[torch.nn.Module, L.LightningModule],
        clean_dataset: Union[str, torch.utils.data.Dataset],
        adversarial_dir: str,
        adversarial_label: int,
        trainer: Union[L.Trainer, BaseTrainer],
        adversarial_transform: Optional[Callable] = None,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        trainer_fit_kwargs: Optional[dict] = None,
        batch_size: int = 8,
    ):
        """Train the model on the mixed dataset.

        Parameters
        ----------
        model: Union[torch.nn.Module, L.LightningModule]
            Model to be used for the benchmark.
        clean_dataset: Union[str, torch.utils.data.Dataset]
            Clean dataset to be used for the benchmark. If a string is passed, it should be a HuggingFace dataset.
        adversarial_dir: str
            Directory containing the adversarial dataset. Typically consists of the same class of objects (e.g. images
            of the same class).
        adversarial_label: int
            The label to be used for the adversarial dataset.
        trainer: Union[L.Trainer, BaseTrainer]
            Trainer to be used for training the model. Can be a Lightning Trainer or a `BaseTrainer`.
        dataset_split: str, optional
            The dataset split, only used for HuggingFace datasets, by default "train".
         dataset_transform : Optional[Callable], optional
             Transform to be applied to the dataset, by default None
        val_dataset: Optional[torch.utils.data.Dataset], optional
            Validation dataset to be used for the benchmark, by default None
        trainer_fit_kwargs: Optional[dict], optional
            Additional keyword arguments for the trainer's fit method, by default None
        seed: int, optional
            Seed for reproducibility, by default 27
        batch_size: int, optional
            Batch size that is used for training, by default 8

        Returns
        -------

        Raises
        ------
        ValueError
            If the model is not a LightningModule and the trainer is a Lightning Trainer.
        ValueError
            If the model is not a torch.nn.Module and the trainer is a BaseTrainer.
        ValueError
            If the trainer is neither a Lightning Trainer nor a BaseTrainer.

        """
        adversarial_dataset = SingleClassImageDataset(
            root=adversarial_dir, label=adversarial_label, transform=adversarial_transform
        )

        self.mixed_dataset = torch.utils.data.ConcatDataset([adversarial_dataset, clean_dataset])
        self.adversarial_label = adversarial_label
        self.adversarial_indices = [1 for i in range(ds_len(adversarial_dataset))] + [
            0 for i in range(ds_len(clean_dataset))
        ]
        self.mixed_train_dl = torch.utils.data.DataLoader(self.mixed_dataset, batch_size=batch_size)

        if val_dataset is not None:
            self.val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
        else:
            self.val_dl = None

        self.model = copy.deepcopy(model)

        trainer_fit_kwargs = trainer_fit_kwargs or {}

        if isinstance(trainer, L.Trainer):
            if not isinstance(self.model, L.LightningModule):
                raise ValueError("Model should be a LightningModule if Trainer is a Lightning Trainer")

            trainer.fit(
                model=self.model,
                train_dataloaders=self.mixed_train_dl,
                val_dataloaders=self.val_dl,
                **trainer_fit_kwargs,
            )

        elif isinstance(trainer, BaseTrainer):
            if not isinstance(self.model, torch.nn.Module):
                raise ValueError("Model should be a torch.nn.Module if Trainer is a BaseTrainer")

            trainer.fit(
                model=self.model,
                train_dataloaders=self.mixed_train_dl,
                val_dataloaders=self.val_dl,
                **trainer_fit_kwargs,
            )

        else:
            raise ValueError("Trainer should be a Lightning Trainer or a BaseTrainer")

    @property
    def bench_state(self):
        """
        Returns the state of the benchmark.

        Returns
        -------
        dict
            The state dictionary of the benchmark.
        """
        return {
            "model": self.model,
            "mixed_dataset": self.dataset_str,
            "adversarial_indices": self.adversarial_indices,
        }

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
            clean_dataset=bench_state["train_dataset"],
            adversarial_dir=bench_state["adversarial_dir"],
            adversarial_label=bench_state["adversarial_label"],
            batch_size=batch_size,
        )

    @classmethod
    def assemble(
        cls,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        adversarial_indices: Union[List[int], torch.Tensor],
        dataset_split: str = "train",
        *args,
        **kwargs,
    ):
        """
        Assembles the benchmark from the given components.

        Parameters
        ----------
        model: Union[torch.nn.Module, L.LightningModule]
            Model to be used for the benchmark.
        clean_dataset: Union[str, torch.utils.data.Dataset]
            Clean dataset to be used for the benchmark. If a string is passed, it should be a HuggingFace dataset.
        adversarial_dir: str
            Directory containing the adversarial dataset. Typically consists of the same class of objects (e.g. images
        adversarial_label: int
            The label to be used for the adversarial dataset.
        dataset_split: str, optional
            The dataset split, only used for HuggingFace datasets, by default "train".
        dataset_transform: Optional[Callable], optional
            Transform to be applied to the dataset, by default None
        args: Any
            Additional positional arguments.
        kwargs: Any
            Additional keyword arguments.

        Returns
        -------

        """

        obj = cls()
        obj.model = model
        obj.mixed_dataset = train_dataset
        obj.adversarial_indices = adversarial_indices
        obj.set_devices(model)

        return obj

    def evaluate(
        self,
        adversarial_expl_dir: str,
        explainer_cls: type,
        adversarial_label: int,
        adversarial_transform: Optional[Callable] = None,
        expl_kwargs: Optional[dict] = None,
        use_predictions: bool = True,
        batch_size: int = 8,
        *args,
        **kwargs,
    ):
        """
        Evaluates the benchmark using a given explanation method.

        Parameters
        ----------
        adversarial_expl_dir: str
            Directory containing the adversarial examples to be explained.
        explainer_cls: type
            The explanation method to be used.
        expl_kwargs: Optional[dict], optional
            Additional keyword arguments for the explanation method, by default None.
        use_predictions: bool, optional
            Whether to use the model's predictions as the target for the explanation method, by default False.
        batch_size: int, optional
            Batch size to be used for evaluation, by default 8.
        args: Any
            Additional positional arguments.
        kwargs: Any
            Additional keyword arguments.

        Returns
        -------

        """
        expl_kwargs = expl_kwargs or {}
        explainer = explainer_cls(model=self.model, train_dataset=self.mixed_dataset, **expl_kwargs)

        adversarial_expl_ds = SingleClassImageDataset(
            root=adversarial_expl_dir, label=adversarial_label, transform=adversarial_transform
        )
        adversarial_expl_dl = torch.utils.data.DataLoader(adversarial_expl_ds, batch_size=batch_size)

        metric = MixedDatasetsMetric(
            model=self.model,
            train_dataset=self.mixed_dataset,
            adversarial_indices=self.adversarial_indices,
        )

        pbar = tqdm(adversarial_expl_dl)
        n_batches = len(adversarial_expl_ds)

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

        return metric.compute()
