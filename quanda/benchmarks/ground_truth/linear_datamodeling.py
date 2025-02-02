"""Benchmark for the Linear Datamodeling Score metric."""

import logging
from typing import Callable, Optional, Union, List, Any

import lightning as L
import torch

from quanda.benchmarks.base import Benchmark
from quanda.benchmarks.resources import (
    load_module_from_bench_state,
    sample_transforms,
)
from quanda.benchmarks.resources.modules import bench_load_state_dict
from quanda.metrics.ground_truth.linear_datamodeling import (
    LinearDatamodelingMetric,
)
from quanda.utils.functions import CorrelationFnLiterals
from quanda.utils.training import BaseTrainer

logger = logging.getLogger(__name__)


class LinearDatamodeling(Benchmark):
    """Benchmark for the Linear Datamodeling Score metric.

    The LDS measures how well a data attribution method can predict the effect
    of retraining a model on different subsets of the training data. It
    computes the correlation between the model’s output when retrained on
    subsets of the data and the attribution method's predictions of those
    outputs.

    References
    ----------
    1) Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc,
    and Aleksander Mądry. (2023). "TRAK: attributing model behavior at scale".
    In Proceedings of the 40th International Conference on Machine Learning"
    (ICML'23), Vol. 202. JMLR.org, Article 1128, (27074–27113).

    2) https://github.com/MadryLab/trak/

    """

    name: str = "Linear Datamodeling Score"
    eval_args: list = ["explanations", "test_data", "test_targets"]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initialize the `LinearDatamodeling` benchmark.

        This initializer is not used directly, instead,
        the `generate` or the `assemble` methods should be used.
        Alternatively, `download` can be used to load a precomputed benchmark.
        """
        super().__init__()

        self.model: Union[torch.nn.Module, L.LightningModule]

        self.train_dataset: torch.utils.data.Dataset
        self.eval_dataset: torch.utils.data.Dataset
        self.dataset_transform: Optional[Callable]
        self.m: int
        self.alpha: float
        self.trainer: Union[L.Trainer, BaseTrainer]
        self.trainer_fit_kwargs: Optional[dict]
        self.cache_dir: str
        self.model_id: str

        self.use_predictions: bool
        self.correlation_fn: Union[Callable, CorrelationFnLiterals]
        self.seed: int
        self.subset_ids: Optional[List[List[int]]]
        self.pretrained_models: Optional[List[torch.nn.Module]]

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
        args : Any
            Variable length argument list.
        kwargs : Any
            Arbitrary keyword arguments.

        """
        obj = cls()
        bench_state = obj._get_bench_state(
            name, cache_dir, device, *args, **kwargs
        )

        eval_dataset = obj._build_eval_dataset(
            dataset_str=bench_state["dataset_str"],
            eval_indices=bench_state["eval_test_indices"],
            transform=sample_transforms[bench_state["dataset_transform"]],
            dataset_split=bench_state["test_split_name"],
        )
        dataset_transform = sample_transforms[bench_state["dataset_transform"]]
        module = load_module_from_bench_state(bench_state, device)

        return obj.assemble(
            model=module,
            checkpoints=bench_state["checkpoints_binary"],
            checkpoints_load_func=bench_load_state_dict,
            train_dataset=bench_state["dataset_str"],
            eval_dataset=eval_dataset,
            m=bench_state["m"],
            cache_dir=bench_state["cache_dir"],
            model_id=bench_state["model_id"],
            alpha=bench_state["alpha"],
            trainer=bench_state["trainer"],
            trainer_fit_kwargs=bench_state["trainer_fit_kwargs"],
            correlation_fn=bench_state["correlation_fn"],
            seed=bench_state["seed"],
            use_predictions=bench_state["use_predictions"],
            subset_ids=bench_state["subset_ids"],
            pretrained_models=bench_state["pretrained_models"],
            dataset_transform=dataset_transform,
        )

    @classmethod
    def assemble(
        cls,
        model: torch.nn.Module,
        train_dataset: Union[str, torch.utils.data.Dataset],
        eval_dataset: torch.utils.data.Dataset,
        trainer: Union[L.Trainer, BaseTrainer],
        cache_dir: str,
        model_id: str,
        m: int = 100,
        alpha: float = 0.5,
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        trainer_fit_kwargs: Optional[dict] = None,
        dataset_transform: Optional[Callable] = None,
        correlation_fn: Union[Callable, CorrelationFnLiterals] = "spearman",
        seed: int = 42,
        use_predictions: bool = True,
        dataset_split: str = "train",
        subset_ids: Optional[List[List[int]]] = None,
        pretrained_models: Optional[List[torch.nn.Module]] = None,
        *args,
        **kwargs,
    ):
        """Assembles the benchmark from existing components.

        Parameters
        ----------
        model : torch.nn.Module
            The model used to generate attributions.
        train_dataset : Union[str, torch.utils.data.Dataset]
            The training dataset used to train `model`. If a string is passed,
            it should be a HuggingFace dataset name.
        eval_dataset : torch.utils.data.Dataset
            The evaluation dataset to be used for the benchmark.
        trainer : Union[L.Trainer, BaseTrainer]
            Trainer to be used for training the models on different subsets.
            Can be a Lightning Trainer or a `BaseTrainer`.
        cache_dir : str
            Directory to be used for caching. This directory will be used to
            save checkpoints of models trained on different subsets of the
            training data.
        model_id : str
            Identifier for the model, to be used in naming cached checkpoints.
        m : int, optional
            Number of subsets to be used for training the models, by default
            100.
        alpha : float, optional
            Percentage of datapoints to be used for training the models, by
            default 0.5.
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s), defaults to None.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file, takes
            (model, checkpoint path) as two arguments, by default None.
        trainer_fit_kwargs : Optional[dict], optional
            Additional keyword arguments to be passed to the `fit` method of
            the trainer, by default None.
        dataset_transform : Optional[Callable], optional
            Transform to be applied to the dataset, by default None.
        correlation_fn : Union[Callable, CorrelationFnLiterals], optional
            Correlation function to be used for the evaluation.
        seed : int, optional
            Seed to be used for the evaluation, by default 42.
        use_predictions : bool, optional
            Whether to use model predictions or the true test labels for the
            evaluation, defaults to False.
        dataset_split : str, optional
            The dataset split to use, by default "train". Only used if
            `train_dataset` is a string.
        subset_ids : Optional[List[List[int]]], optional
            A list of pre-defined subset indices, by default None.
        pretrained_models : Optional[List[torch.nn.Module]], optional
            A list of pre-trained models for each subset, by default None.
        args : Any
            Additional arguments.
        kwargs : Any
            Additional keyword arguments.

        """
        obj = cls()
        obj._assemble_common(
            model=model,
            eval_dataset=eval_dataset,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
            use_predictions=use_predictions,
        )
        obj.subset_ids = subset_ids
        obj.pretrained_models = pretrained_models
        obj.correlation_fn = correlation_fn
        obj.trainer = trainer
        obj.m = m
        obj.alpha = alpha
        obj.trainer_fit_kwargs = trainer_fit_kwargs
        obj.seed = seed
        obj.cache_dir = cache_dir
        obj.model_id = model_id
        obj.train_dataset = obj._process_dataset(
            train_dataset,
            transform=dataset_transform,
            dataset_split=dataset_split,
        )
        # this sets the function to the default value
        obj.checkpoints_load_func = None

        return obj

    generate = assemble

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
            Batch size to be used for the evaluation, defaults to 8

        Returns
        -------
        dict
            Dictionary containing the evaluation results.

        """
        explainer = self._prepare_explainer(
            dataset=self.train_dataset,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
        )

        metric = LinearDatamodelingMetric(
            model=self.model,
            checkpoints=self.checkpoints,
            train_dataset=self.train_dataset,
            alpha=self.alpha,
            m=self.m,
            trainer=self.trainer,
            trainer_fit_kwargs=self.trainer_fit_kwargs,
            cache_dir=self.cache_dir,
            model_id=self.model_id,
            correlation_fn=self.correlation_fn,
            seed=self.seed,
            batch_size=batch_size,
            subset_ids=self.subset_ids,
            pretrained_models=self.pretrained_models,
        )
        return self._evaluate_dataset(
            eval_dataset=self.eval_dataset,
            explainer=explainer,
            metric=metric,
            batch_size=batch_size,
        )
