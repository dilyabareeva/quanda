import logging
from typing import Callable, Optional, Union

import torch
import torch.utils
from tqdm import tqdm

from quanda.benchmarks.base import Benchmark
from quanda.benchmarks.resources import (
    load_module_from_bench_state,
    sample_transforms,
)
from quanda.metrics.heuristics import TopKOverlapMetric

logger = logging.getLogger(__name__)


class TopKOverlap(Benchmark):
    """
    Benchmark for top-k overlap heuristic.

    TODO: remove USES PREDICTED LABELS https://arxiv.org/pdf/2006.04528
    """

    name: str = "Top-K Overlap"

    def __init__(
        self,
        *args,
        **kwargs,
    ):
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
        data_transform: Optional[Callable] = None,
        top_k: int = 1,
        use_predictions: bool = True,
        dataset_split: str = "train",
        *args,
        **kwargs,
    ):
        """
        This method should generate all the benchmark components and persist them in the instance.
        """

        logger.info(f"Generating {TopKOverlap.name} benchmark components based on passed arguments...")

        obj = cls(train_dataset)
        obj.set_devices(model)
        obj.eval_dataset = eval_dataset
        obj.train_dataset = obj.process_dataset(train_dataset, transform=data_transform, dataset_split=dataset_split)
        obj.top_k = top_k
        obj.use_predictions = use_predictions
        obj.model = model

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
        """
        This method should load the benchmark components from a file and persist them in the instance.
        """
        obj = cls()
        bench_state = obj._get_bench_state(name, cache_dir, device, *args, **kwargs)

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
        )

    @classmethod
    def assemble(
        cls,
        model: torch.nn.Module,
        train_dataset: Union[str, torch.utils.data.Dataset],
        eval_dataset: torch.utils.data.Dataset,
        data_transform: Optional[Callable] = None,
        top_k: int = 1,
        use_predictions: bool = True,
        dataset_split: str = "train",
        *args,
        **kwargs,
    ):
        """
        This method should assemble the benchmark components from arguments and persist them in the instance.
        """
        obj = cls()
        obj.set_devices(model)
        obj.train_dataset = obj.process_dataset(train_dataset, transform=data_transform, dataset_split=dataset_split)
        obj.eval_dataset = eval_dataset
        obj.use_predictions = use_predictions
        obj.model = model
        obj.top_k = top_k
        obj.set_devices(model)

        return obj

    def evaluate(
        self,
        explainer_cls: type,
        expl_kwargs: Optional[dict] = None,
        batch_size: int = 8,
    ):
        self.model.eval()

        expl_kwargs = expl_kwargs or {}
        explainer = explainer_cls(model=self.model, train_dataset=self.train_dataset, **expl_kwargs)

        expl_dl = torch.utils.data.DataLoader(self.eval_dataset, batch_size=batch_size)

        metric = TopKOverlapMetric(model=self.model, train_dataset=self.train_dataset, top_k=self.top_k)

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
            metric.update(explanations)

        return metric.compute()
