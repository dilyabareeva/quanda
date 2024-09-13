from typing import Callable, Optional, Union

import torch
from tqdm import tqdm

from quanda.benchmarks.base import Benchmark
from quanda.metrics.heuristics.model_randomization import (
    ModelRandomizationMetric,
)
from quanda.utils.functions import CorrelationFnLiterals


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
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.model: torch.nn.Module
        self.train_dataset: torch.utils.data.Dataset

    @classmethod
    def generate(
        cls,
        train_dataset: Union[str, torch.utils.data.Dataset],
        model: torch.nn.Module,
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
        dataset_split : str, optional
            The dataset split to use, by default "train". Only used if `train_dataset` is a string.
        """

        obj = cls()
        obj.set_devices(model)
        obj.train_dataset = obj.process_dataset(train_dataset, dataset_split)
        obj.model = model

        return obj

    @property
    def bench_state(self):
        """
        Returns the benchmark state as a dictionary.

        Returns
        -------
        dict
            The benchmark state.
        """
        return {
            "model": self.model,
            "train_dataset": self.dataset_str,  # ok this probably won't work, but that's the idea
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

        return cls.assemble(model=bench_state["model"], train_dataset=bench_state["train_dataset"])

    @classmethod
    def assemble(
        cls,
        model: torch.nn.Module,
        train_dataset: Union[str, torch.utils.data.Dataset],
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
        dataset_split : str, optional
            The dataset split, only used for HuggingFace datasets, by default "train".
        """
        obj = cls()
        obj.model = model
        obj.train_dataset = obj.process_dataset(train_dataset, dataset_split)
        obj.set_devices(model)

        return obj

    def evaluate(
        self,
        expl_dataset: torch.utils.data.Dataset,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        use_predictions: bool = False,
        correlation_fn: Union[Callable, CorrelationFnLiterals] = "spearman",
        seed: int = 42,
        cache_dir: str = "./cache",
        model_id: str = "default_model_id",
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
        expl_dl = torch.utils.data.DataLoader(expl_dataset, batch_size=batch_size)

        metric = ModelRandomizationMetric(
            model=self.model,
            train_dataset=self.train_dataset,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
            correlation_fn=correlation_fn,
            seed=seed,
            model_id=model_id,
            cache_dir=cache_dir,
        )
        pbar = tqdm(expl_dl)
        n_batches = len(expl_dl)

        for i, (input, labels) in enumerate(pbar):
            pbar.set_description("Metric evaluation, batch %d/%d" % (i + 1, n_batches))

            input, labels = input.to(self.device), labels.to(self.device)

            if use_predictions:
                with torch.no_grad():
                    output = self.model(input)
                    targets = output.argmax(dim=-1)
            else:
                targets = labels

            metric.explain_update(test_data=input, explanation_targets=targets)

        return metric.compute()
