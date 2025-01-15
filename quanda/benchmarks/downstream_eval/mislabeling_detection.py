"""Benchmark for noisy label detection."""

import logging
import os
import warnings
from typing import Callable, Dict, List, Optional, Union, Any

import lightning as L
import torch
import torch.utils

from quanda.benchmarks.base import Benchmark
from quanda.metrics.downstream_eval import MislabelingDetectionMetric
from quanda.utils.datasets.transformed.label_flipping import (
    LabelFlippingDataset,
)
from quanda.utils.training.trainer import BaseTrainer

logger = logging.getLogger(__name__)


class MislabelingDetection(Benchmark):
    # TODO: remove ALL PAPERS USE SELF-INFLUENCE? OTHERWISE WE CAN USE
    #  PREDICTIONS
    """Benchmark for noisy label detection.

    This benchmark generates a dataset with mislabeled samples, and trains a
    model on it. Afterward, it evaluates the effectiveness of a given data
    attributor for detecting the mislabeled examples using
    ´quanda.metrics.downstream_eval.MislabelingDetectionMetric´.

    This is done by computing a cumulative detection curve (as described in the
    below references) and calculating the AUC following Kwon et al. (2024).

    References
    ----------
    1) Koh, P. W., & Liang, P. (2017). Understanding black-box predictions via
    influence functions. In International Conference on Machine Learning
    (pp. 1885-1894). PMLR.

    2) Yeh, C.-K., Kim, J., Yen, I. E., Ravikumar, P., & Dhillon, I. S. (2018).
    Representer point selection for explaining deep neural networks. In
    Advances in Neural Information Processing Systems (Vol. 31).

    3) Pruthi, G., Liu, F., Sundararajan, M., & Kale, S. (2020). Estimating
    training data influence by tracing gradient descent. In Advances in Neural
    Information Processing Systems (Vol. 33, pp. 19920-19930).

    4) Picard, A. M., Vigouroux, D., Zamolodtchikov, P., Vincenot, Q., Loubes,
    J.-M., & Pauwels, E. (2022). Leveraging influence functions for dataset
    exploration and cleaning. In 11th European Congress on Embedded Real-Time
    Systems (ERTS 2022) (pp. 1-8). Toulouse, France.

    5) Kwon, Y., Wu, E., Wu, K., & Zou, J. (2024). DataInf: Efficiently
    estimating data influence in LoRA-tuned LLMs and diffusion models. In The
    Twelfth International Conference on Learning Representations (pp. 1-8).

    """

    name: str = "Mislabeling Detection"
    eval_args = ["test_data", "test_labels", "explanations"]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initialize the Mislabeling Detection benchmark.

        This initializer is not used directly, instead,
        the `generate` or the `assemble` methods should be used.
        Alternatively, `download` can be used to load a precomputed benchmark.
        """
        super().__init__()

        self.model: Union[torch.nn.Module, L.LightningModule]

        self.base_dataset: torch.utils.data.Dataset
        self.eval_dataset: torch.utils.data.Dataset
        self.mislabeling_dataset: LabelFlippingDataset
        self.dataset_transform: Optional[Callable]
        self.mislabeling_indices: List[int]
        self.mislabeling_labels: Dict[int, int]
        self.mislabeling_train_dl: torch.utils.data.DataLoader
        self.mislabeling_val_dl: Optional[torch.utils.data.DataLoader]
        self.p: float
        self.global_method: Union[str, type] = "self-influence"
        self.n_classes: int
        self.use_predictions: bool

    @classmethod
    def generate(
        cls,
        model: Union[torch.nn.Module, L.LightningModule],
        base_dataset: Union[str, torch.utils.data.Dataset],
        n_classes: int,
        trainer: Union[L.Trainer, BaseTrainer],
        cache_dir: str,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        use_predictions: bool = True,
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
        """Generate the benchmark by specifying parameters.

        This module handles the dataset creation and model training on the
        label-poisoned dataset. The evaluation can then be run using the
        `evaluate` method.

        Parameters
        ----------
        model : Union[torch.nn.Module, L.LightningModule]
            Model to be used for the benchmark.
            Note that a new model will be trained on the label-poisoned
            dataset.
        base_dataset : Union[str, torch.utils.data.Dataset]
            Vanilla training dataset to be used for the benchmark. If a string
            is passed, it should be a HuggingFace dataset.
        n_classes : int
            Number of classes in the dataset.
        trainer : Union[L.Trainer, BaseTrainer]
            Trainer to be used for training the model. Can be a Lightning
            Trainer or a `BaseTrainer`.
        cache_dir : str
            Directory to store the generated benchmark components.
        eval_dataset : Optional[torch.utils.data.Dataset]
            Dataset to be used for the evaluation.
            This is only used if `global_method` is not "self-influence", by
            default None.
            Original papers use the self-influence method to reach a global
            ranking of the data,
            instead of using aggregations of generated local explanations.
        use_predictions : bool, optional
            Whether to use the model's predictions for the evaluation.
            This is only used if `global_method` is not "self-influence", by
            default True.
            Original papers use the self-influence method to reach a global
            ranking of the data,
            instead of using aggregations of generated local explanations.
        dataset_split : str, optional
            The dataset split, only used for HuggingFace datasets, by default
            "train".
        dataset_transform : Optional[Callable], optional
            Transform to be applied to the dataset, by default None
        val_dataset : Optional[torch.utils.data.Dataset], optional
            Validation dataset to be used for the benchmark, by default None
        global_method : Union[str, type], optional
            Method to generate a global ranking from local explainer.
            It can be a subclass of
            `quanda.explainers.aggregators.BaseAggregator` or "self-influence".
            Defaults to "self-influence".
        p : float, optional
            The probability of mislabeling per sample, by default 0.3.
        trainer_fit_kwargs : Optional[dict], optional
            Additional keyword arguments for the trainer's fit method, by
            default None.
        seed : int, optional
            Seed for reproducibility, by default 27.
        batch_size : int, optional
            Batch size that is used for training, by default 8.
        args: Any
            Additional arguments.
        kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        MislabelingDetection
            The benchmark instance.

        """
        logger.info(
            f"Generating {MislabelingDetection.name} benchmark components "
            f"based on passed arguments..."
        )
        if global_method != "self-influence":
            assert eval_dataset is not None, (
                "MislabelingDetection should have "
                "global_method='self-influence' or eval_dataset should be "
                "given."
            )

        obj = cls()

        save_dir = os.path.join(cache_dir, "model_mislabeling_detection.pth")
        base_dataset = obj._process_dataset(
            base_dataset, transform=None, dataset_split=dataset_split
        )
        mislabeling_dataset = LabelFlippingDataset(
            dataset=base_dataset,
            p=p,
            dataset_transform=dataset_transform,
            n_classes=n_classes,
            seed=seed,
        )

        mislabeling_labels = mislabeling_dataset.mislabeling_labels

        obj = obj.assemble(
            model=model,
            base_dataset=base_dataset,
            n_classes=n_classes,
            mislabeling_labels=mislabeling_labels,
            mislabeling_dataset=mislabeling_dataset,
            checkpoints=[save_dir],
            checkpoints_load_func=None,
            eval_dataset=eval_dataset,
            use_predictions=use_predictions,
            dataset_split=dataset_split,
            dataset_transform=dataset_transform,
            global_method=global_method,
            batch_size=batch_size,
        )

        obj.model = obj._train_model(
            model=model,
            trainer=trainer,
            train_dataset=obj.mislabeling_dataset,
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
        mislabeling_labels: Dict[int, int],
        mislabeling_dataset: Optional[LabelFlippingDataset] = None,
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        use_predictions: bool = True,
        dataset_split: str = "train",
        dataset_transform: Optional[Callable] = None,
        global_method: Union[str, type] = "self-influence",
        batch_size: int = 8,
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
        mislabeling_labels : Dict[int, int]
            Dictionary containing indices as keys and new labels as values.
        mislabeling_dataset : Optional[torch.utils.data.Dataset], optional
            Dataset with mislabeled samples, by default None.
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s), defaults to None.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file, takes
            (model, checkpoint path) as two arguments, by default None.
        eval_dataset : Optional[torch.utils.data.Dataset]
            Dataset to be used for the evaluation by default None.
        use_predictions : bool, optional
            Whether to use the model's predictions for the evaluation.
            This is only used if `global_method` is not "self-influence",
            by default True. Original papers use the self-influence method to
            reach a global ranking of the data, instead of using aggregations
            of generated local explanations.
        dataset_split : str, optional
            The dataset split, only used for HuggingFace datasets, by default
            "train".
        dataset_transform : Optional[Callable], optional
            Transform to be applied to the dataset, by default None
        global_method : Union[str, type], optional
            Method to generate a global ranking from local explainer.
            It can be a subclass of
            `quanda.explainers.aggregators.BaseAggregator` or "self-influence".
            Defaults to "self-influence".
        batch_size : int, optional
            Batch size that is used for training, by default 8.
        checkpoint_paths : Optional[List[str]], optional
            List of paths to the checkpoints. This parameter is only used for
            downloaded benchmarks, by default None.
        args: Any
            Additional arguments.
        kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        MislabelingDetection
            The benchmark instance.

        """
        if global_method != "self-influence":
            assert eval_dataset is not None, (
                "MislabelingDetection should have "
                "global_method='self-influence' or eval_dataset should be "
                "given."
            )

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
        obj.global_method = global_method
        obj.n_classes = n_classes
        mislabeling_indices = (
            list(mislabeling_labels.keys())
            if mislabeling_labels is not None
            else None
        )

        if mislabeling_dataset is not None:
            warnings.warn(
                "mislabeling_dataset was passed, mislabeling_labels "
                "will be ignored."
            )
            obj.mislabeling_dataset = mislabeling_dataset
        else:
            obj.mislabeling_dataset = LabelFlippingDataset(
                dataset=obj._process_dataset(
                    base_dataset, transform=None, dataset_split=dataset_split
                ),
                dataset_transform=dataset_transform,
                transform_indices=mislabeling_indices,
                n_classes=n_classes,
                mislabeling_labels=mislabeling_labels,
            )

        obj.mislabeling_indices = obj.mislabeling_dataset.transform_indices
        obj.mislabeling_labels = obj.mislabeling_dataset.mislabeling_labels

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
            Batch size to be used for the evaluation, defaults to 8.

        Returns
        -------
        dict
            Dictionary containing the evaluation results.

        """
        explainer = self._prepare_explainer(
            dataset=self.mislabeling_dataset,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
        )

        if self.global_method != "self-influence":
            if self.eval_dataset is None:
                raise ValueError(
                    "eval_dataset should be given for non-self-influence "
                    "methods."
                )

            mislabeling_expl_ds = LabelFlippingDataset(
                dataset=self.eval_dataset,
                dataset_transform=self.dataset_transform,
                n_classes=self.n_classes,
                p=0.0,
            )

            metric = MislabelingDetectionMetric.aggr_based(
                model=self.model,
                train_dataset=self.mislabeling_dataset,
                mislabeling_indices=self.mislabeling_indices,
                aggregator_cls=self.global_method,
            )

            return self._evaluate_dataset(
                eval_dataset=mislabeling_expl_ds,
                explainer=explainer,
                metric=metric,
                batch_size=batch_size,
            )

        else:
            metric = MislabelingDetectionMetric.self_influence_based(
                model=self.model,
                checkpoints=self.checkpoints,
                checkpoints_load_func=self.checkpoints_load_func,
                train_dataset=self.mislabeling_dataset,
                mislabeling_indices=self.mislabeling_indices,
                explainer_cls=explainer_cls,
                expl_kwargs=expl_kwargs,
            )

            return metric.compute()
