"""Class Detection benchmark."""

import logging
from typing import Callable, List, Optional, Any

import torch

from quanda.benchmarks.base import Benchmark
from quanda.metrics.downstream_eval import ClassDetectionMetric

logger = logging.getLogger(__name__)


class ClassDetection(Benchmark):
    """Benchmark for class detection task.

    This benchmark evaluates the effectiveness of an attribution method in
    detecting the class of a test sample from its highest attributed training
    point. Intuitively, a good attribution method should assign the highest
    attribution to the class of the test sample, as argued in Hanawa et al.
    (2021) and Kwon et al. (2024).

    References
    ----------
    1) Hanawa, K., Yokoi, S., Hara, S., & Inui, K. (2021). Evaluation of
    similarity-based explanations. In International Conference on Learning
    Representations.

    2) Kwon, Y., Wu, E., Wu, K., Zou, J., (2024). DataInf: Efficiently
    Estimating Data Influence in LoRA-tuned LLMs and Diffusion Models. The
    Twelfth International Conference on Learning Representations.

    """

    # TODO: remove USES PREDICTED LABELS https://arxiv.org/pdf/2006.04528
    name: str = "Class Detection"
    eval_args = ["test_targets", "explanations"]

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        """Initialize the Class Detection benchmark.

        This initializer is not used directly, instead,
        the `generate` or the `assemble` methods should be used.
        Alternatively, `download` can be used to load a precomputed benchmark.
        """
        super().__init__()

        self.model: torch.nn.Module
        self.device: str
        self.train_dataset: torch.utils.data.Dataset
        self.eval_dataset: torch.utils.data.Dataset
        self.use_predictions: bool
        self.checkpoints: List[str]
        self.checkpoints_load_func: Callable[..., Any]

    @classmethod
    def from_config(
        cls,
        config: dict,
        load_meta_from_disk: bool = True,
        offline: bool = False,
        device: str = "cpu",
    ):
        """Initialize the benchmark from a dictionary.

        Parameters
        ----------
        config : dict
            Dictionary containing the configuration.
        load_meta_from_disk : str
            Loads dataset metadata from disk if True, otherwise generates it,
            default True.
        offline : bool
            If True, the model is not downloaded, default False.
        device: str, optional
            Device to use for the evaluation, by default "cpu".

        """
        obj = super().from_config(config, load_meta_from_disk, offline, device)
        obj.use_predictions = config.get("use_predictions", True)
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
            dataset=self.train_dataset,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
        )

        metric = ClassDetectionMetric(
            model=self.model,
            checkpoints=self.checkpoints,
            train_dataset=self.train_dataset,
            checkpoints_load_func=self.checkpoints_load_func,
        )

        return self._evaluate_dataset(
            eval_dataset=self.eval_dataset,
            explainer=explainer,
            metric=metric,
            batch_size=batch_size,
        )
