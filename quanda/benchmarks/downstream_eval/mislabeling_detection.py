"""Benchmark for noisy label detection."""
import copy
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
from quanda.utils.datasets.transformed.metadata import LabelFlippingMetadata
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
        self.eval_dataset: torch.nn.utils.Dataset
        self.train_dataset: LabelFlippingDataset
        self.device: str

        self.global_method: Union[str, type] = "self-influence"
        self.use_predictions: bool = True
        self.checkpoints: Optional[List[str]]
        self.checkpoints_load_func: Optional[Callable[..., Any]]

    @classmethod
    def from_config(cls, config: dict, load_meta_from_disk: bool = True,
                    device: str = "cpu"):
        """Initialize the benchmark from a dictionary.

        Parameters
        ----------
        config : dict
            Dictionary containing the configuration.
        load_meta_from_disk : str
            Loads dataset metadata from disk if True, otherwise generates it, 
            default True.
        device: str, optional
            Device to use for the evaluation, by default "cpu".

        """
        obj = super().from_config(config, load_meta_from_disk, device)
        obj.global_method = config.get("global_method", "self-influence")
        obj.use_predictions = config.get("use_predictions", True)
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
            dataset=self.train_dataset,
            explainer_cls=explainer_cls,
            expl_kwargs=expl_kwargs,
        )
        
        if isinstance(self.eval_dataset, LabelFlippingDataset):
            raise ValueError(
                "Evaluation dataset in Mislabeling Metric should not have "
                "flipped labels."
            )

        if not isinstance(self.train_dataset, LabelFlippingDataset):
            raise ValueError(
                "Training dataset in Mislabeling Metric should have flipped "
                "labels."
            )

        if self.global_method != "self-influence":
            if self.eval_dataset is None:
                raise ValueError(
                    "eval_dataset should be given for non-self-influence "
                    "methods."
                )

            metric = MislabelingDetectionMetric.aggr_based(
                model=self.model,
                train_dataset=self.train_dataset,
                mislabeling_indices=self.train_dataset.metadata.transform_indices,
                aggregator_cls=self.global_method,
            )

            return self._evaluate_dataset(
                eval_dataset=self.eval_dataset,
                explainer=explainer,
                metric=metric,
                batch_size=batch_size,
            )

        else:
            metric = MislabelingDetectionMetric.self_influence_based(
                model=self.model,
                checkpoints=self.checkpoints,
                checkpoints_load_func=self.checkpoints_load_func,
                train_dataset=self.train_dataset,
                mislabeling_indices=self.train_dataset.metadata.transform_indices,
                explainer_cls=explainer_cls,
                expl_kwargs=expl_kwargs,
            )

            return metric.compute()
