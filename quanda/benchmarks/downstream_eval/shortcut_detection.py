"""Shortcut Detection Benchmark."""

import os
import warnings
from typing import Callable, List, Optional, Union, Any

import lightning as L
import torch

from quanda.benchmarks.base import Benchmark
from quanda.metrics.downstream_eval.shortcut_detection import (
    ShortcutDetectionMetric,
)
from quanda.utils.datasets.transformed.sample import (
    SampleTransformationDataset,
)
from quanda.utils.training.trainer import BaseTrainer


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

        self.base_dataset: torch.utils.data.Dataset
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
        base_dataset: Union[str, torch.utils.data.Dataset],
        n_classes: int,
        eval_dataset: torch.utils.data.Dataset,
        shortcut_cls: int,
        trainer: Union[L.Trainer, BaseTrainer],
        sample_fn: Callable,
        cache_dir: str,
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
        base_dataset : Union[str, torch.utils.data.Dataset]
            The vanilla training dataset to be used for the benchmark.
            If a string is passed, it should be a HuggingFace dataset.
        n_classes : int
            Number of classes in the dataset.
        eval_dataset : torch.utils.data.Dataset
            Dataset to be used for the evaluation.
        shortcut_cls : int
            The class to add triggers to.
        trainer : Union[L.Trainer, BaseTrainer]
            Trainer to be used for training the model.
        sample_fn : Callable
            Function to add triggers to samples of the dataset.
        cache_dir : str
            Directory to store the generated benchmark components.
        filter_by_prediction : bool, optional
            Whether to filter the test samples to only calculate the metric on
            those samples, where the shortcut class
            is predicted, by default True.
        filter_by_class: bool, optional
            Whether to filter the test samples to only calculate the metric on
            those samples, where the shortcut class
            is not assigned as the class, by default False.
        use_predictions : bool, optional
            Whether to use the model's predictions for the evaluation, by
            default True.
        dataset_split : str, optional
            Split used for HuggingFace datasets, by default "train".
        dataset_transform : Optional[Callable], optional
            Default transform of the dataset, by default None.
        val_dataset : Optional[torch.utils.data.Dataset], optional
            Validation dataset to use during training, by default None.
        p : float, optional
            The probability of poisoning with the trigger per sample, by
            default 0.3.
        trainer_fit_kwargs : Optional[dict], optional
            Keyword arguments to supply the trainer, by default None.
        seed : int, optional
            seed for reproducibility, by default 27.
        batch_size : int, optional
            Batch size to use during training, by default 8.
        args: Any
            Additional arguments.
        kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        ShortcutDetection
            An instance of the ShortcutDetection benchmark.

        """
        obj = cls()

        save_dir = os.path.join(cache_dir, "model_shortcut_detection.pth")
        base_dataset = obj._process_dataset(
            base_dataset, transform=None, dataset_split=dataset_split
        )
        shortcut_dataset = SampleTransformationDataset(
            dataset=base_dataset,
            p=p,
            dataset_transform=dataset_transform,
            cls_idx=shortcut_cls,
            n_classes=n_classes,
            sample_fn=sample_fn,
            seed=seed,
        )
        shortcut_indices = shortcut_dataset.transform_indices

        obj = obj.assemble(
            model=model,
            base_dataset=base_dataset,
            n_classes=n_classes,
            eval_dataset=eval_dataset,
            sample_fn=sample_fn,
            shortcut_cls=shortcut_cls,
            shortcut_indices=shortcut_indices,
            shortcut_dataset=shortcut_dataset,
            checkpoints=[save_dir],
            checkpoints_load_func=None,
            filter_by_prediction=filter_by_prediction,
            filter_by_class=filter_by_class,
            use_predictions=use_predictions,
            dataset_split=dataset_split,
            dataset_transform=dataset_transform,
            checkpoint_paths=None,
        )

        if val_dataset:
            val_dataset = SampleTransformationDataset(
                dataset=val_dataset,
                dataset_transform=obj.dataset_transform,
                p=obj.p,
                cls_idx=shortcut_cls,
                sample_fn=sample_fn,
                n_classes=obj.n_classes,
            )

        obj.model = obj._train_model(
            model=model,
            trainer=trainer,
            train_dataset=obj.shortcut_dataset,
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
        eval_dataset: torch.utils.data.Dataset,
        sample_fn: Callable,
        shortcut_cls: int,
        shortcut_indices: List[int],
        shortcut_dataset: Optional[SampleTransformationDataset] = None,
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        filter_by_prediction: bool = True,
        filter_by_class: bool = False,
        use_predictions: bool = True,
        dataset_split: str = "train",
        dataset_transform: Optional[Callable] = None,
        checkpoint_paths: Optional[List[str]] = None,
        *args,
        **kwargs,
    ):
        """Assembles the benchmark from existing components.

        Parameters
        ----------
        model : Union[torch.nn.Module, L.LightningModule]
            Model to be used for the benchmark. This model should be trained on
            the mislabeled dataset.
        base_dataset : Union[str, torch.utils.data.Dataset]
            Training dataset to be used for the benchmark. If a string is
            passed, it should be a HuggingFace dataset.
        n_classes : int
            Number of classes in the dataset.
        eval_dataset : torch.utils.data.Dataset
            Dataset to be used for the evaluation.
        sample_fn : Callable
            Function to add triggers to samples of the dataset.
        shortcut_cls : int
            The class to use.
        shortcut_indices : List[int]
            Binary list of indices to poison.
        shortcut_dataset : Optional[SampleTransformationDataset], optional
            Dataset with the shortcut, by default None.
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s), defaults to None.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file, takes
            (model, checkpoint path) as two arguments, by default None.
        filter_by_prediction : bool, optional
            Whether to filter the test samples to only calculate the metric on
            those samples, where the shortcut class
            is predicted, by default True
        filter_by_class: bool, optional
            Whether to filter the test samples to only calculate the metric on
            those samples, where the shortcut class
            is not assigned as the class, by default False
        use_predictions : bool, optional
            Whether to use the model's predictions for the evaluation, by
            default True.
        dataset_split : str, optional
            The dataset split, only used for HuggingFace datasets, by default
            "train".
        dataset_transform : Optional[Callable], optional
            Transform to be applied to the dataset, by default None.
        checkpoint_paths : Optional[List[str]], optional
            List of paths to the checkpoints. This parameter is only used for
            downloaded benchmarks, by default None.
        args: Any
            Additional arguments.
        kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        ShortcutDetection
            The benchmark instance.

        """
        obj = cls()
        obj._assemble_common(
            model=model,
            eval_dataset=eval_dataset,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
            use_predictions=use_predictions,
        )
        obj.base_dataset = obj._process_dataset(
            base_dataset,
            transform=dataset_transform,
            dataset_split=dataset_split,
        )
        obj.dataset_transform = dataset_transform
        obj.n_classes = n_classes
        obj.filter_by_prediction = filter_by_prediction
        obj.filter_by_class = filter_by_class

        if shortcut_dataset is not None:
            warnings.warn(
                "shortcut_dataset is not None. Ignoring other shortcut "
                "parameters."
            )
            obj.shortcut_dataset = shortcut_dataset
        else:
            obj.shortcut_dataset = SampleTransformationDataset(
                dataset=obj._process_dataset(
                    base_dataset, transform=None, dataset_split=dataset_split
                ),
                cls_idx=shortcut_cls,
                dataset_transform=dataset_transform,
                sample_fn=sample_fn,
                n_classes=n_classes,
                transform_indices=shortcut_indices,
            )
        obj.shortcut_cls = shortcut_cls
        obj.shortcut_indices = obj.shortcut_dataset.transform_indices
        obj.sample_fn = sample_fn
        obj._checkpoint_paths = checkpoint_paths

        return obj

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
            dataset=self.shortcut_dataset,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
        )

        shortcut_expl_ds = SampleTransformationDataset(
            dataset=self.eval_dataset,
            dataset_transform=self.dataset_transform,
            n_classes=self.n_classes,
            sample_fn=self.sample_fn,
            p=1.0,
        )

        metric = ShortcutDetectionMetric(
            model=self.model,
            checkpoints=self.checkpoints,
            checkpoints_load_func=self.checkpoints_load_func,
            train_dataset=self.shortcut_dataset,
            shortcut_indices=self.shortcut_indices,
            shortcut_cls=self.shortcut_cls,
            filter_by_prediction=self.filter_by_prediction,
            filter_by_class=self.filter_by_class,
        )
        return self._evaluate_dataset(
            eval_dataset=shortcut_expl_ds,
            explainer=explainer,
            metric=metric,
            batch_size=batch_size,
        )
