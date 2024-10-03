import logging
import os
from typing import Callable, List, Optional, Union

import torch
from tqdm import tqdm

from quanda.benchmarks.base import Benchmark
from quanda.benchmarks.resources import (
    load_module_from_bench_state,
    sample_transforms,
)
from quanda.metrics.heuristics.model_randomization import (
    ModelRandomizationMetric,
)
from quanda.utils.functions import CorrelationFnLiterals

logger = logging.getLogger(__name__)


class ModelRandomization(Benchmark):
    # TODO: remove UNKNOWN IF PREDICTED LABELS ARE USED https://arxiv.org/pdf/2006.04528
    """
    Benchmark for the model randomization heuristic.

    This benchmark is used to evaluate the dependence of the attributions on the model parameters.

    References
    ----------
    1) Hanawa, K., Yokoi, S., Hara, S., & Inui, K. (2021). Evaluation of similarity-based explanations. In International
    Conference on Learning Representations.

    2) Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim, B. (2018). Sanity checks for saliency
    maps. In Advances in Neural Information Processing Systems (Vol. 31).
    """

    name: str = "Model Randomization"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """
        Initializer for the Model Randomization benchmark.

        This initializer is not used directly, instead,
        the `generate` or the `assemble` methods should be used.
        Alternatively, `download` can be used to load a precomputed benchmark.
        """
        super().__init__()

        self.model: torch.nn.Module
        self.train_dataset: torch.utils.data.Dataset
        self.eval_dataset: torch.utils.data.Dataset
        self.use_predictions: bool
        self.correlation_fn: Union[Callable, CorrelationFnLiterals]
        self.seed: int

    @classmethod
    def generate(
        cls,
        train_dataset: Union[str, torch.utils.data.Dataset],
        eval_dataset: torch.utils.data.Dataset,
        model: torch.nn.Module,
        data_transform: Optional[Callable] = None,
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
        data_transform : Optional[Callable], optional
            Transform to be applied to the dataset, by default None.
        dataset_split : str, optional
            The dataset split to use, by default "train". Only used if `train_dataset` is a string.
        correlation_fn : Union[Callable, CorrelationFnLiterals], optional
            Correlation function to be used for the evaluation.
            Can be "spearman" or "kendall", or a callable.
            Defaults to "spearman".
        use_predictions: bool
            Whether to use the model's predictions for generating attributions. Defaults to True.
        seed : int, optional
            Seed to be used for the evaluation, by default 42.

        Returns
        -------
        ModelRandomization
            The benchmark instance.
        """

        logger.info(f"Generating {ModelRandomization.name} benchmark components based on passed arguments...")

        obj = cls()
        obj.set_devices(model)
        obj.train_dataset = obj.process_dataset(train_dataset, transform=data_transform, dataset_split=dataset_split)
        obj.eval_dataset = eval_dataset
        obj.correlation_fn = correlation_fn
        obj.seed = seed
        obj.use_predictions = use_predictions
        obj.model = model

        return obj

    @classmethod
    def download(cls, name: str, cache_dir: str, device: str, *args, **kwargs):
        """
        This method loads precomputed benchmark components from a file and creates an instance
        from the state dictionary.

        Parameters
        ----------
        name : str
            Name of the benchmark to be loaded.
        cache_dir : str
            Directory to store the downloaded benchmark components.
        device : str
            Device to load the model on.

        Returns
        -------
        ModelRandomization
            The benchmark instance.
        """
        obj = cls()
        bench_state = obj._get_bench_state(name, cache_dir, device, *args, **kwargs)

        checkpoint_paths = []
        for ckpt_name, ckpt in zip(bench_state["checkpoints"], bench_state["checkpoints_binary"]):
            save_path = os.path.join(cache_dir, ckpt_name)
            torch.save(ckpt, save_path)
            checkpoint_paths.append(save_path)

        eval_dataset = obj.build_eval_dataset(
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
            use_predictions=bench_state["use_predictions"],
            data_transform=dataset_transform,
            checkpoint_paths=checkpoint_paths,
        )

    @classmethod
    def assemble(
        cls,
        model: torch.nn.Module,
        train_dataset: Union[str, torch.utils.data.Dataset],
        eval_dataset: torch.utils.data.Dataset,
        data_transform: Optional[Callable] = None,
        correlation_fn: Union[Callable, CorrelationFnLiterals] = "spearman",
        seed: int = 42,
        use_predictions: bool = True,
        dataset_split: str = "train",
        checkpoint_paths: Optional[List[str]] = None,
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
        data_transform : Optional[Callable], optional
            Transform to be applied to the dataset, by default None.
        dataset_split : str, optional
            The dataset split, only used for HuggingFace datasets, by default "train".
        correlation_fn : Union[Callable, CorrelationFnLiterals], optional
            Correlation function to be used for the evaluation.
            Can be "spearman" or "kendall", or a callable.
            Defaults to "spearman".
        use_predictions: bool
            Whether to use the model's predictions for generating attributions. Defaults to True.
        seed : int, optional
            Seed to be used for the evaluation, by default 42.
        checkpoint_paths : Optional[List[str]], optional
            List of paths to the checkpoints. This parameter is only used for downloaded benchmarks, by default None.

        Returns
        -------
        ModelRandomization
            The benchmark instance.
        """
        obj = cls()
        obj.model = model
        obj.eval_dataset = eval_dataset
        obj.use_predictions = use_predictions
        obj.correlation_fn = correlation_fn
        obj.seed = seed
        obj.train_dataset = obj.process_dataset(train_dataset, transform=data_transform, dataset_split=dataset_split)
        obj.set_devices(model)
        obj._checkpoint_paths = checkpoint_paths

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
            Batch size to be used for the evaluation, default to 8.

        Returns
        -------
        Dict[str, float]
            Dictionary containing the evaluation results.
        """

        self.model.eval()

        expl_kwargs = expl_kwargs or {}
        explainer = explainer_cls(model=self.model, train_dataset=self.train_dataset, **expl_kwargs)
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
