"""Top-K Cardinality benchmark module."""

import logging
import os
from typing import Callable, List, Optional, Union, Any

import torch
import torch.utils
from tqdm import tqdm

from quanda.benchmarks.base import Benchmark
from quanda.benchmarks.resources import (
    load_module_from_bench_state,
    sample_transforms,
)
from quanda.benchmarks.resources.modules import bench_load_state_dict
from quanda.metrics.heuristics import TopKCardinalityMetric
from quanda.utils.common import load_last_checkpoint

logger = logging.getLogger(__name__)


class TopKCardinality(Benchmark):
    # TODO: remove USES PREDICTED LABELS https://arxiv.org/pdf/2006.04528
    """Benchmark for the Top-K Cardinality heuristic.

    This benchmark evaluates the dependence of the attributions on the test
    samples being attributed. The cardinality of the union of top-k attributed
    training samples is computed. A higher cardinality indicates variance in
    the attributions, which indicates dependence on the test samples.

    References
    ----------
    1) Barshan, Elnaz, Marc-Etienne Brunet, and Gintare Karolina Dziugaite.
    (2020). Relatif: Identifying explanatory training samples via relative
    influence. International Conference on Artificial Intelligence and
    Statistics. PMLR.

    """

    name: str = "Top-K Cardinality"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initialize the Top-K Cardinality benchmark.

        This initializer is not used directly, instead,
        the `generate` or the `assemble` methods should be used.
        Alternatively, `download` can be used to load a precomputed benchmark.
        """
        super().__init__()

        self.model: torch.nn.Module

        self.train_dataset: torch.utils.data.Dataset
        self.eval_dataset: torch.utils.data.Dataset
        self.use_predictions: bool
        self.top_k: int

    @classmethod
    def generate(
        cls,
        train_dataset: Union[str, torch.utils.data.Dataset],
        model: torch.nn.Module,
        eval_dataset: torch.utils.data.Dataset,
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        data_transform: Optional[Callable] = None,
        top_k: int = 1,
        use_predictions: bool = True,
        dataset_split: str = "train",
        *args,
        **kwargs,
    ):
        """Generate the benchmark by specifying parameters.

        The evaluation can then be run using the `evaluate` method.

        Parameters
        ----------
        train_dataset : Union[str, torch.utils.data.Dataset]
            The training dataset used to train the model.
        model : torch.nn.Module
            The model to be evaluated.
        eval_dataset : torch.utils.data.Dataset
            The evaluation dataset.
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s), defaults to None.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file, takes
            (model, checkpoint path) as two arguments, by default None.
        data_transform : Optional[Callable], optional
            The transform to be applied to the dataset, by default None
        top_k : int, optional
            The number of top-k samples to consider, by default 1
        use_predictions : bool, optional
            Whether to use the model's predictions for the evaluation, by
            default True
        dataset_split : str, optional
            _description_, by default "train"
        args: Any
            Additional arguments.
        kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        TopKCardinality
            The benchmark instance.

        """
        logger.info(
            f"Generating {TopKCardinality.name} benchmark components based on "
            f"passed arguments..."
        )

        obj = cls(train_dataset)
        obj._set_devices(model)
        obj.eval_dataset = eval_dataset
        obj.train_dataset = obj._process_dataset(
            train_dataset,
            transform=data_transform,
            dataset_split=dataset_split,
        )
        obj.top_k = top_k
        obj.use_predictions = use_predictions
        obj.model = model
        obj.checkpoints = checkpoints
        obj.checkpoints_load_func = checkpoints_load_func

        return obj

    @classmethod
    def download(
        cls,
        name: str,
        cache_dir: str,
        device: str,
        *args,
        **kwargs,
    ):
        """Download a precomputed benchmark from URL.

        Load precomputed benchmark components from a file and creates an
        instance from the state dictionary.

        Parameters
        ----------
        name : str
            Name of the benchmark to be loaded.
        cache_dir : str
            Directory where the benchmark components are stored.
        device : str
            Device to be used for the model.
        args: Any
            Additional arguments.
        kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        TopKCardinality
            The benchmark instance.

        """
        obj = cls()
        bench_state = obj._get_bench_state(
            name, cache_dir, device, *args, **kwargs
        )

        checkpoint_paths = []
        for ckpt_name, ckpt in zip(
            bench_state["checkpoints"], bench_state["checkpoints_binary"]
        ):
            save_path = os.path.join(cache_dir, ckpt_name)
            torch.save(ckpt, save_path)
            checkpoint_paths.append(save_path)

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
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        data_transform: Optional[Callable] = None,
        top_k: int = 1,
        use_predictions: bool = True,
        dataset_split: str = "train",
        checkpoint_paths: Optional[List[str]] = None,
        *args,
        **kwargs,
    ):
        """Assembles the benchmark from existing components.

        Parameters
        ----------
        model : torch.nn.Module
            The model to be evaluated.
        train_dataset : Union[str, torch.utils.data.Dataset]
            The training dataset used to train the model.
        eval_dataset : torch.utils.data.Dataset
            The dataset to be used for the evaluation.
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s), defaults to None.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file, takes
            (model, checkpoint path) as two arguments, by default None.
        data_transform : Optional[Callable], optional
            The transform to be applied to the dataset, by default None.
        top_k : int, optional
            The number of top-k samples to consider, by default 1.
        use_predictions : bool, optional
            Whether to use the model's predictions for the evaluation, by
            default True.
        dataset_split : str, optional
            The dataset split, by default "train", only used for HuggingFace
            datasets.
        checkpoint_paths : Optional[List[str]], optional
            List of paths to the checkpoints. This parameter is only used for
            downloaded benchmarks, by default None.
        args: Any
            Additional arguments.
        kwargs: Any
            Additional keyword arguments.

        Returns
        -------
        TopKCardinality
            The benchmark instance.

        """
        obj = cls()
        obj._set_devices(model)
        obj.train_dataset = obj._process_dataset(
            train_dataset,
            transform=data_transform,
            dataset_split=dataset_split,
        )
        obj.eval_dataset = eval_dataset
        obj.use_predictions = use_predictions
        obj.model = model
        obj.checkpoints = checkpoints
        obj.checkpoints_load_func = checkpoints_load_func
        obj.top_k = top_k
        obj._set_devices(model)
        obj._checkpoint_paths = checkpoint_paths

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
        load_last_checkpoint(
            model=self.model,
            checkpoints=self.checkpoints,
            checkpoints_load_func=self.checkpoints_load_func,
        )
        self.model.eval()

        expl_kwargs = expl_kwargs or {}
        explainer = explainer_cls(
            model=self.model,
            checkpoints=self.checkpoints,
            train_dataset=self.train_dataset,
            checkpoints_load_func=self.checkpoints_load_func,
            **expl_kwargs,
        )

        expl_dl = torch.utils.data.DataLoader(
            self.eval_dataset, batch_size=batch_size
        )

        metric = TopKCardinalityMetric(
            model=self.model,
            checkpoints=self.checkpoints,
            train_dataset=self.train_dataset,
            checkpoints_load_func=self.checkpoints_load_func,
            top_k=self.top_k,
        )

        pbar = tqdm(expl_dl)
        n_batches = len(expl_dl)

        for i, (input, labels) in enumerate(pbar):
            pbar.set_description(
                "Metric evaluation, batch %d/%d" % (i + 1, n_batches)
            )

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
            metric.update(explanations=explanations)

        return metric.compute()
