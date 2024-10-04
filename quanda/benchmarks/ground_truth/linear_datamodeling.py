import logging
from typing import Callable, Optional, Union

import lightning as L
import torch
from tqdm import tqdm

from quanda.benchmarks.base import Benchmark
from quanda.benchmarks.resources import (
    load_module_from_bench_state,
    sample_transforms,
)
from quanda.metrics.ground_truth.linear_datamodeling import (
    LinearDatamodelingMetric,
)
from quanda.utils.functions import CorrelationFnLiterals
from quanda.utils.training import BaseTrainer

logger = logging.getLogger(__name__)


class LinearDatamodeling(Benchmark):
    """
    Benchmark for the Linear Datamodeling Score metric.

    The LDS measures how well a data attribution method can predict the effect of retraining
    a model on different subsets of the training data. It computes the correlation between
    the model’s output when retrained on subsets of the data and the attribution method's predictions
    of those outputs.

    References
    ----------
    1) Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc,
        and Aleksander Mądry. (2023). "TRAK: attributing model behavior at scale".
        In Proceedings of the 40th International Conference on Machine Learning" (ICML'23), Vol. 202.
        JMLR.org, Article 1128, (27074–27113).

    2) https://github.com/MadryLab/trak/
    """

    name: str = "Linear Datamodeling Score"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        Initializer for the `LinearDatamodeling` benchmark.

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

    @classmethod
    def generate(
        cls,
        train_dataset: Union[str, torch.utils.data.Dataset],
        eval_dataset: torch.utils.data.Dataset,
        model: torch.nn.Module,
        trainer: Union[L.Trainer, BaseTrainer],
        cache_dir: str,
        model_id: str,
        data_transform: Optional[Callable] = None,
        correlation_fn: Union[Callable, CorrelationFnLiterals] = "spearman",
        m: int = 100,
        alpha: float = 0.5,
        trainer_fit_kwargs: Optional[dict] = None,
        seed: int = 42,
        use_predictions: bool = True,
        dataset_split: str = "train",
        *args,
        **kwargs,
    ):
        """
        This method generates the benchmark components and creates an instance.

        Parameters
        ----------
        train_dataset : Union[str, torch.utils.data.Dataset]
            The training dataset used to train `model`. If a string is passed, it should be a HuggingFace dataset name.
        model : torch.nn.Module
            The model used to generate attributions.
        eval_dataset : torch.utils.data.Dataset
            The evaluation dataset to be used for the benchmark.
        trainer : Union[L.Trainer, BaseTrainer]
            Trainer to be used for training the models on different subsets. Can be a Lightning Trainer or a `BaseTrainer`.
        cache_dir : str
            Directory to be used for caching. This directory will be used to save checkpoints of models
            trained on different subsets of the training data.
        model_id : str
            Identifier for the model, to be used in naming cached checkpoints.
        data_transform : Optional[Callable], optional
            Transform to be applied to the dataset, by default None.
        correlation_fn : Union[Callable, CorrelationFnLiterals], optional
            Correlation function to be used for the evaluation.
        m: int, optional
            Number of subsets to be used for training the models, by default 100.
        alpha: float, optional
            Percentage of datapoints to be used for training the models, by default 0.5.
        trainer_fit_kwargs : Optional[dict], optional
            Additional keyword arguments to be passed to the `fit` method of the trainer, by default None.
        seed : int, optional
            Seed to be used for the evaluation, by default 42.
        use_predictions : bool, optional
            Whether to use model predictions or the true test labels for the evaluation, defaults to False.
        dataset_split : str, optional
            The dataset split to use, by default "train". Only used if `train_dataset` is a string.
        """

        logger.info(f"Generating {LinearDatamodeling.name} benchmark components based on passed arguments...")

        obj = cls()
        obj._set_devices(model)
        obj.train_dataset = obj._process_dataset(train_dataset, transform=data_transform, dataset_split=dataset_split)
        obj.eval_dataset = eval_dataset
        obj.correlation_fn = correlation_fn
        obj.seed = seed
        obj.use_predictions = use_predictions
        obj.model = model
        obj.trainer = trainer
        obj.m = m
        obj.alpha = alpha
        obj.trainer_fit_kwargs = trainer_fit_kwargs
        obj.cache_dir = cache_dir
        obj.model_id = model_id

        return obj

    @classmethod
    def download(cls, name: str, cache_dir: str, device: str, *args, **kwargs):
        # add (cache_dir, model_id) and (checkpoints_dir, subsets, model_id) to the cache_dir
        """
        This method loads precomputed benchmark components from a file and creates an instance from the state dictionary.

        Parameters
        ----------
        name : str
            Name of the benchmark to be loaded.
        cache_dir : str
            Directory to store the downloaded benchmark components.
        device : str
            Device to load the model on.
        """
        obj = cls()
        bench_state = obj._get_bench_state(name, cache_dir, device, *args, **kwargs)

        eval_dataset = obj._build_eval_dataset(
            dataset_str=bench_state["dataset_str"],
            eval_indices=bench_state["eval_test_indices"],
            transform=sample_transforms[bench_state["dataset_transform"]],
            dataset_split="test",
        )
        dataset_transform = sample_transforms[bench_state["dataset_transform"]]
        module = load_module_from_bench_state(bench_state, device)

        return obj.assemble(
            model=module,
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
            data_transform=dataset_transform,
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
        trainer_fit_kwargs: Optional[dict] = None,
        data_transform: Optional[Callable] = None,
        correlation_fn: Union[Callable, CorrelationFnLiterals] = "spearman",
        seed: int = 42,
        use_predictions: bool = True,
        dataset_split: str = "train",
        *args,
        **kwargs,
    ):
        """
        Assembles the benchmark from existing components.

        Parameters
        ----------
        model : torch.nn.Module
            The model used to generate attributions.
        train_dataset : Union[str, torch.utils.data.Dataset]
            The training dataset used to train `model`. If a string is passed, it should be a HuggingFace dataset name.
        eval_dataset : torch.utils.data.Dataset
            The evaluation dataset to be used for the benchmark.
        trainer : Union[L.Trainer, BaseTrainer]
            Trainer to be used for training the models on different subsets. Can be a Lightning Trainer or a `BaseTrainer`.
        cache_dir : str
            Directory to be used for caching. This directory will be used to save checkpoints of models
            trained on different subsets of the training data.
        model_id : str
            Identifier for the model, to be used in naming cached checkpoints.
        m: int, optional
            Number of subsets to be used for training the models, by default 100.
        alpha: float, optional
            Percentage of datapoints to be used for training the models, by default 0.5.
        trainer_fit_kwargs : Optional[dict], optional
            Additional keyword arguments to be passed to the `fit` method of the trainer, by default None.
        data_transform : Optional[Callable], optional
            Transform to be applied to the dataset, by default None.
        correlation_fn : Union[Callable, CorrelationFnLiterals], optional
            Correlation function to be used for the evaluation.
        seed : int, optional
            Seed to be used for the evaluation, by default 42.
        use_predictions : bool, optional
            Whether to use model predictions or the true test labels for the evaluation, defaults to False.
        dataset_split : str, optional
            The dataset split to use, by default "train". Only used if `train_dataset` is a string.
        """
        obj = cls()
        obj.model = model
        obj.eval_dataset = eval_dataset
        obj.use_predictions = use_predictions
        obj.correlation_fn = correlation_fn
        obj.trainer = trainer
        obj.m = m
        obj.alpha = alpha
        obj.trainer_fit_kwargs = trainer_fit_kwargs
        obj.seed = seed
        obj.cache_dir = cache_dir
        obj.model_id = model_id
        obj.train_dataset = obj._process_dataset(train_dataset, transform=data_transform, dataset_split=dataset_split)
        obj._set_devices(model)

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
            Additional keyword arguments for the explainer, by default None.
        batch_size : int, optional
            Batch size to be used for the evaluation, defaults to 8

        Returns
        -------
        dict
            Dictionary containing the evaluation results.
        """

        self.model.eval()

        expl_kwargs = expl_kwargs or {}
        explainer = explainer_cls(model=self.model, train_dataset=self.train_dataset, **expl_kwargs)
        expl_dl = torch.utils.data.DataLoader(self.eval_dataset, batch_size=batch_size)

        metric = LinearDatamodelingMetric(
            model=self.model,
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
                test_tensor=input,
                targets=targets,
            )

            metric.update(explanations=explanations, test_tensor=input, explanation_targets=targets)

        return metric.compute()
