"""Mixed Datasets benchmark module."""

import logging
import os
from typing import Callable, List, Optional, Union, Any

import lightning as L
import torch

from quanda.benchmarks.base import Benchmark
from quanda.metrics.heuristics.mixed_datasets import MixedDatasetsMetric
from quanda.utils.common import ds_len
from quanda.utils.datasets import SingleClassImageDataset
from quanda.utils.training.trainer import BaseTrainer

logger = logging.getLogger(__name__)


class MixedDatasets(Benchmark):
    # TODO: remove FILTER BY "CORRECT" PREDICTION FOR BACKDOOR implied
    #  https://arxiv.org/pdf/2201.10055
    """Mixed Datasets Benchmark.

    Evaluates the performance of a given data attribution estimation method in
    identifying adversarial examples in a classification task.

    The training dataset is assumed to consist of a "clean" and "adversarial"
    subsets, whereby the number of samples in the clean dataset is
    significantly larger than the number of samples in the adversarial dataset.
    All adversarial samples are labeled with one label from the clean dataset.
    The evaluation is based on the area under the precision-recall curve
    (AUPRC), which quantifies the ranking of the influence of adversarial
    relative to clean samples. AUPRC is chosen because it provides better
    insight into performance in highly-skewed classification tasks where
    false positives are common.

    Unlike the original implementation, we only employ a single trained model,
    but we aggregate the AUPRC scores across
    multiple test samples.

    References
    ----------
    1) Hammoudeh, Z., & Lowd, D. (2022). Identifying a training-set attack's
    target using renormalized influence estimation. In Proceedings of the 2022
    ACM SIGSAC Conference on Computer and Communications Security
    (pp. 1367-1381).

    """

    name: str = "Mixed Datasets"
    eval_args: list = ["explanations", "test_data", "test_labels"]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initialize the Mixed Datasets benchmark.

        This initializer is not used directly, instead,
        the `generate` or the `assemble` methods should be used.
        Alternatively, `download` can be used to load a precomputed benchmark.
        """
        super().__init__()

        self.model: Union[torch.nn.Module, L.LightningModule]

        self.base_dataset: torch.utils.data.Dataset
        self.eval_dataset: torch.utils.data.Dataset
        self.mixed_dataset: torch.utils.data.Dataset
        self.adversarial_indices: List[int]
        self.use_predictions: bool
        self.adversarial_label: int
        self.filter_by_prediction: bool
        self.cache_dir: str

    @classmethod
    def generate(
        cls,
        model: Union[torch.nn.Module, L.LightningModule],
        base_dataset: Union[str, torch.utils.data.Dataset],
        eval_dataset: torch.utils.data.Dataset,
        adversarial_dir: str,
        adversarial_label: int,
        adv_train_indices: List[int],
        trainer: Union[L.Trainer, BaseTrainer],
        cache_dir: str,
        dataset_transform: Optional[Callable] = None,
        use_predictions: bool = True,
        filter_by_prediction: bool = True,
        dataset_split: str = "train",
        adversarial_transform: Optional[Callable] = None,
        val_dataset: Optional[torch.utils.data.Dataset] = None,
        trainer_fit_kwargs: Optional[dict] = None,
        batch_size: int = 8,
        *args,
        **kwargs,
    ):
        """Generate the benchmark with passed components.

        This module handles the dataset creation and model training on the
        mixed dataset. The evaluation can then be run using the `evaluate`
        method.

        Parameters
        ----------
        model: Union[torch.nn.Module, L.LightningModule]
            Model to be used for the benchmark.
        base_dataset: Union[str, torch.utils.data.Dataset]
            Clean dataset to be used for the benchmark. If a string is passed,
            it should be a HuggingFace dataset. If a torch Dataset is passed,
            every item of the dataset is a tuple of the form (input, label).
        eval_dataset: torch.utils.data.Dataset
            The dataset containing the adversarial examples used for
            evaluation. They should belong to the same dataset and the same
            class as the samples in the adversarial dataset.
        adversarial_dir: str
            Path to directory containing the adversarial dataset. Typically
            consists of the same class of objects (e.g. images of the same
            class).
        adversarial_label: int
            The label to be used for the adversarial dataset.
        adv_train_indices: List[int]
            List of indices of the adversarial dataset used for training.
        trainer: Union[L.Trainer, BaseTrainer]
            Trainer to be used for training the model. Can be a Lightning
            Trainer or a `BaseTrainer`.
        cache_dir: str
            Directory to store the generated benchmark.
        dataset_transform: Optional[Callable], optional
            Transform to be applied to the clean dataset, by default None.
        use_predictions: bool, optional
            Whether to use the model's predictions for generating attributions.
            Defaults to True.
        filter_by_prediction: bool, optional
            Whether to filter the adversarial examples to only use correctly
            predicted test samples. Defaults to True.
        dataset_split: str, optional
            The dataset split, only used for HuggingFace datasets, by default
            "train".
        adversarial_transform : Optional[Callable], optional
             Transform to be applied to the adversarial dataset, by default
             None.
        val_dataset: Optional[torch.utils.data.Dataset], optional
            Validation dataset to be used for the benchmark, by default None.
        trainer_fit_kwargs: Optional[dict], optional
            Additional keyword arguments for the trainer's fit method, by
            default None.
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

        Raises
        ------
        ValueError
            If the model is not a LightningModule and the trainer is a
            Lightning Trainer.
        ValueError
            If the model is not a torch.nn.Module and the trainer is a
            BaseTrainer.
        ValueError
            If the trainer is neither a Lightning Trainer nor a BaseTrainer.

        """
        logger.info(
            f"Generating {MixedDatasets.name} benchmark components based on "
            f"passed arguments..."
        )

        save_dir = os.path.join(cache_dir, "model_mixed_datasets.pth")

        obj = cls()
        obj = obj.assemble(
            model=model,
            eval_dataset=eval_dataset,
            base_dataset=base_dataset,
            adversarial_dir=adversarial_dir,
            adversarial_label=adversarial_label,
            adv_train_indices=adv_train_indices,
            checkpoints=[save_dir],
            checkpoints_load_func=None,
            dataset_transform=dataset_transform,
            use_predictions=use_predictions,
            filter_by_prediction=filter_by_prediction,
            adversarial_transform=adversarial_transform,
            dataset_split=dataset_split,
        )

        obj.model = obj._train_model(
            model=model,
            trainer=trainer,
            train_dataset=obj.mixed_dataset,
            val_dataset=val_dataset,  # TODO: validate, why just val_dataset?
            save_dir=save_dir,
            trainer_fit_kwargs=trainer_fit_kwargs,
            batch_size=batch_size,
        )
        return obj

    @classmethod
    def assemble(
        cls,
        model: Union[torch.nn.Module, L.LightningModule],
        eval_dataset: torch.utils.data.Dataset,
        base_dataset: Union[str, torch.utils.data.Dataset],
        adversarial_dir: str,
        adversarial_label: int,
        adv_train_indices: List[int],
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        dataset_transform: Optional[Callable] = None,
        use_predictions: bool = True,
        filter_by_prediction: bool = True,
        adversarial_transform: Optional[Callable] = None,
        dataset_split: str = "train",
        *args,
        **kwargs,
    ):
        """Assembles the benchmark from the given components.

        Parameters
        ----------
        model: Union[torch.nn.Module, L.LightningModule]
            Model to be used for the benchmark.
        eval_dataset: torch.utils.data.Dataset
            The dataset containing the adversarial examples used for
            evaluation. They should belong to the same dataset and the same
            class as the samples in the adversarial dataset.
        base_dataset: Union[str, torch.utils.data.Dataset]
            Clean dataset to be used for the benchmark. If a string is passed,
            it should be a HuggingFace dataset.
        adversarial_dir: str
            Path to the adversarial dataset of a single class.
        adversarial_label: int
            The label to be used for the adversarial dataset.
        adv_train_indices: List[int]
            List of indices of the adversarial dataset used for training.
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s), defaults to None.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file, takes
            (model, checkpoint path) as two arguments, by default None.
        dataset_transform: Optional[Callable], optional
            Transform to be applied to the clean dataset, by default None.
        use_predictions: bool, optional
            Whether to use the model's predictions for generating attributions.
            Defaults to True.
        filter_by_prediction: bool, optional
            Whether to filter the adversarial examples to only use correctly
            predicted test samples. Defaults to True.
        adversarial_transform: Optional[Callable], optional
            Transform to be applied to the adversarial dataset, by default
            None.
        dataset_split: str, optional
            The dataset split, only used for HuggingFace datasets, by default
            "train".
        args: Any
            Additional arguments.
        kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        MixedDatasets
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
        obj.filter_by_prediction = filter_by_prediction
        obj.adversarial_label = adversarial_label

        adversarial_dataset = SingleClassImageDataset(
            root=adversarial_dir,
            label=adversarial_label,
            transform=adversarial_transform,
            indices=adv_train_indices,
        )

        obj.mixed_dataset = torch.utils.data.ConcatDataset(
            [adversarial_dataset, obj.base_dataset]
        )
        obj.adversarial_indices = [1] * ds_len(adversarial_dataset) + [
            0
        ] * ds_len(obj.base_dataset)

        return obj

    def evaluate(
        self,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        batch_size: int = 8,
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

        Returns
        -------
        Dict[str, float]
            Dictionary containing the metric score.

        """
        explainer = self._prepare_explainer(
            dataset=self.mixed_dataset,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
        )

        metric = MixedDatasetsMetric(
            model=self.model,
            checkpoints=self.checkpoints,
            train_dataset=self.mixed_dataset,
            checkpoints_load_func=self.checkpoints_load_func,
            adversarial_indices=self.adversarial_indices,
            filter_by_prediction=self.filter_by_prediction,
            adversarial_label=self.adversarial_label,
        )

        return self._evaluate_dataset(
            eval_dataset=self.eval_dataset,
            explainer=explainer,
            metric=metric,
            batch_size=batch_size,
        )
