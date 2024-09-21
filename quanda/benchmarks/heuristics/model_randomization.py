import logging
from typing import Callable, Optional, Union

import torch
from tqdm import tqdm

from quanda.benchmarks.base import Benchmark
from quanda.metrics.heuristics.model_randomization import (
    ModelRandomizationMetric,
)
from quanda.utils.functions import CorrelationFnLiterals

logger = logging.getLogger(__name__)


class ModelRandomization(Benchmark):
    """
    Benchmark for the model randomization heuristic.

    This benchmark is used to evaluate the dependence of the attributions on the model parameters..

    References
    ----------
    1) Hanawa, K., Yokoi, S., Hara, S., & Inui, K. (2021). Evaluation of similarity-based explanations. In International
    Conference on Learning Representations.

    2) Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). Sanity checks for saliency
    maps. In Advances in Neural Information Processing Systems (Vol. 31).

    TODO: remove UNKNOWN IF PREDICTED LABELS ARE USED https://arxiv.org/pdf/2006.04528
    """

    name: str = "Model Randomization"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.model: torch.nn.Module
        self.train_dataset: torch.utils.data.Dataset
        self.eval_dataset: torch.utils.data.Dataset

    @classmethod
    def generate(
        cls,
        train_dataset: Union[str, torch.utils.data.Dataset],
        eval_dataset: torch.utils.data.Dataset,
        model: torch.nn.Module,
            correlation_fn: Union[Callable, CorrelationFnLiterals] = "spearman",
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
        dataset_split : str, optional
            The dataset split to use, by default "train". Only used if `train_dataset` is a string.
        """

        logger.info(f"Generating {ModelRandomization.name} benchmark components based on passed arguments...")

        obj = cls()
        obj.set_devices(model)
        obj.train_dataset = obj.process_dataset(train_dataset, dataset_split)
        obj.eval_dataset = eval_dataset
        obj.correlation_fn = correlation_fn
        obj.seed = seed
        obj.use_predictions = use_predictions
        obj.model = model

        return obj

    @classmethod
    def download(cls, name: str, eval_dataset: torch.utils.data.Dataset, batch_size: int = 32, *args, **kwargs):
        """
        This method loads precomputed benchmark components from a file and creates an instance from the state dictionary.

        Parameters
        ----------
        name : str
            Name of the benchmark to be loaded.
        eval_dataset : torch.utils.data.Dataset
            The evaluation dataset to be used for the benchmark.
        """
        bench_state = cls.download_bench_state(name)

        return cls.assemble(
            model=bench_state["model"],
            eval_dataset=eval_dataset,
            use_predictions=bench_state["use_predictions"],
            train_dataset=bench_state["train_dataset"],
        )

    @classmethod
    def assemble(
        cls,
        model: torch.nn.Module,
        train_dataset: Union[str, torch.utils.data.Dataset],
        eval_dataset: torch.utils.data.Dataset,
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
        model : Union[torch.nn.Module, L.LightningModule]
            Model to be used for the benchmark. This model should be trained on the mislabeled dataset.
        train_dataset : Union[str, torch.utils.data.Dataset]
            Training dataset to be used for the benchmark. If a string is passed, it should be a HuggingFace dataset.
        eval_dataset : torch.utils.data.Dataset
            Evaluation dataset to be used for the benchmark.
        dataset_split : str, optional
            The dataset split, only used for HuggingFace datasets, by default "train".
        """
        obj = cls()
        obj.model = model
        obj.eval_dataset = eval_dataset
        obj.use_predictions = use_predictions
        obj.correlation_fn = correlation_fn
        obj.seed = seed
        obj.train_dataset = obj.process_dataset(train_dataset, dataset_split)
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
        use_predictions : bool, optional
            Whether to use model predictions or the true test labels for the evaluation, defaults to False
        correlation_fn : Union[Callable, CorrelationFnLiterals], optional
            Correlation function to be used for the evaluation
            Can be "spearman" or "kendall", or a callable.
            Defaults to "spearman"
        batch_size : int, optional
            Batch size to be used for the evaluation, default to 8
        seed : int, optional
            Seed to be used for the evaluation, defaults to 42
        cache_dir : str, optional
            Directory to be used for caching, defaults to "./cache"
        model_id : str, optional
            Identifier for the model, defaults to "default_model_id"
        batch_size : int, optional
            Batch size to be used for the evaluation, defaults to 8

        Returns
        -------
        dict
            Dictionary containing the evaluation results.
        """

        expl_kwargs = expl_kwargs or {}
        explainer = explainer_cls(
            model=self.model, train_dataset=self.train_dataset, **expl_kwargs
        )
        expl_dl = torch.utils.data.DataLoader(self.eval_dataset, batch_size=batch_size)

        metric = ModelRandomizationMetric(
            model=self.model,
            train_dataset=self.train_dataset,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
            correlation_fn=self.correlation_fn,
            seed=self.seed,
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

            metric.update(explanations=explanations, test_data=input, explanation_targets=targets)

        return metric.compute()
