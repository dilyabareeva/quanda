"""Linear Datamodeling Score (LDS) metric ."""

import os
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Union

import lightning as L
import torch
import yaml
from torch.nn.functional import log_softmax
from torch.utils.data import DataLoader

from quanda.metrics.base import Metric
from quanda.utils.common import ds_len
from quanda.utils.functions import CorrelationFnLiterals, correlation_functions
from quanda.utils.training import BaseTrainer


class LinearDatamodelingMetric(Metric):
    """Metric for the Linear Datamodeling Score (LDS).

    The LDS measures how well a data attribution method can predict the effect
    of retraining a model on different subsets of the training data. It
    computes the correlation between the model’s output when retrained on
    subsets of the data and the attribution method's predictions of those
    outputs.

    References
    ----------
    1) Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc,
        and Aleksander Mądry. (2023). "TRAK: attributing model behavior at
        scale". In Proceedings of the 40th International Conference on Machine
        Learning" (ICML'23), Vol. 202. JMLR.org, Article 1128, (27074–27113).

    2) https://github.com/MadryLab/trak/

    """

    def __init__(
        self,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        trainer: Optional[Union[L.Trainer, BaseTrainer]] = None,
        alpha: float = 0.5,
        m: int = 100,
        correlation_fn: Union[Callable, CorrelationFnLiterals] = "spearman",
        trainer_fit_kwargs: Optional[dict] = None,
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable] = None,
        seed: int = 42,
        batch_size: int = 32,
        subset_ids: Optional[Union[List[List[int]], str]] = None,
        subset_ckpt_filenames: Optional[List[str]] = None,
        model_id: Optional[str] = "0",
        cache_dir: str = "./cache",
    ):
        """Initialize the LinearDatamodelingMetric.

        Parameters
        ----------
        model : Union[torch.nn.Module, L.LightningModule]
            The model used to generate attributions.
        train_dataset : torch.utils.data.Dataset
            The training dataset used to train models.
        trainer : Union[L.Trainer, BaseTrainer]
            Trainer object used to fit the model on the sampled subsets.
        alpha : float, optional
            The fraction of the training data to include in each subset, by
            default 0.5.
        m : int, optional
            Number of subsets to sample, by default 100.
        correlation_fn : Union[Callable, CorrelationFnLiterals], optional
            Correlation function to use, by default "spearman". Can be
            "spearman", "kendall", or a callable.
        trainer_fit_kwargs : Optional[dict], optional
            Additional keyword arguments for the trainer, by default None.
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s), defaults to None.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file, takes
            (model, checkpoint path) as two arguments, by default None.
        seed : Optional[int], optional
            Random seed for reproducibility, by default 42.
        batch_size : int, optional
            Batch size for training, by default 32.
        subset_ids : Optional[List[List[int]]], optional
            A list of pre-defined subset indices, by default None.
        subset_ckpt_filenames : Optional[List[torch.nn.Module]], optional
            A list of pre-trained models for each subset, by default None.
        model_id : str
            An identifier for the model, by default "0".
        cache_dir : str
            The cache directory for the checkpoints, by default "./cache".

        """
        super().__init__(
            model=model,
            checkpoints=checkpoints,
            train_dataset=train_dataset,
            checkpoints_load_func=checkpoints_load_func,
        )

        if subset_ids is not None:
            if isinstance(subset_ids, str):
                assert os.path.exists(f"{cache_dir}/{subset_ids}"), (
                    f"No file found at {cache_dir}/{subset_ids}"
                )
                with open(f"{cache_dir}/{subset_ids}", "r") as f:
                    self.subset_ids = yaml.safe_load(f)
            else:
                self.subset_ids = subset_ids

        else:
            self.subset_ids = self.generate_subsets(
                dataset=train_dataset,
                alpha=alpha,
                m=m,
                generator=torch.Generator().manual_seed(seed),
            )
        self.cache_dir = cache_dir
        self.model_id = model_id

        LinearDatamodelingMetric._validate_parameters(
            correlation_fn, subset_ids, subset_ckpt_filenames, trainer
        )

        if (
            isinstance(correlation_fn, str)
            and correlation_fn in correlation_functions
        ):
            self.corr_measure = correlation_functions[correlation_fn]
        elif callable(correlation_fn):
            self.corr_measure = correlation_fn

        self.results: Dict[str, List[torch.Tensor]] = {"scores": []}
        self.m = m
        self.alpha = alpha
        self.trainer = trainer
        self.trainer_fit_kwargs = trainer_fit_kwargs
        self.seed = seed
        self.batch_size = batch_size

        self.generator = None
        if self.seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(self.seed)

        self.subsets = [
            torch.utils.data.Subset(train_dataset, indices)
            for indices in self.subset_ids
        ]
        if subset_ckpt_filenames is None:
            self.subset_ckpt_filenames = self.train_subset_models()
        else:
            # TODO: validate that the checkpoints exist
            self.subset_ckpt_filenames = subset_ckpt_filenames

    @classmethod
    def _validate_parameters(
        cls, correlation_fn, subset_ids, pretrained_models, trainer
    ):
        if not (
            (
                isinstance(correlation_fn, str)
                and correlation_fn in correlation_functions
            )
            or callable(correlation_fn)
        ):
            raise ValueError(
                f"Invalid correlation function: expected one of "
                f"{list(correlation_functions.keys())} or"
                f"a Callable, but got {correlation_fn}."
            )
        if (
            trainer is None
            and pretrained_models is None
            and subset_ids is None
        ):
            raise ValueError(
                "Invalid combination of argumetns."
                "Either trainer should be given, "
                "or both pretrained_models and subset_ids"
                "should be specified."
            )

    @staticmethod
    def generate_subsets(
        dataset: torch.utils.data.Dataset,
        alpha: float,
        m: int,
        generator: Optional[torch.Generator] = None,
    ):
        """Generate subsets of the dataset.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            The dataset to sample subsets from.
        subset_ids : List[List[int]]
            Indices of datapoints for each subset.
        alpha : float
            Fraction of the dataset size to use for each subset.
        m : int
            Number of subsets to sample.
        generator : torch.Generator
            Random generator for reproducibility.

        Returns
        -------
        List[torch.utils.data.Subset]
            A list of m subsets of the training data.

        """
        N = ds_len(dataset)
        subset_size = int(alpha * N)

        subset_ids = []
        for _ in range(m):
            indices = list(
                torch.randperm(N, generator=generator)[:subset_size].tolist()
            )
            subset_ids.append(indices)
        return subset_ids

    def train_subset_models(self) -> List[str]:
        """Train counterfactual model on a subset.

        Returns
        -------
        List[str]
            A list of filenames of the trained counterfactual models.

        Raises
        ------
        ValueError
            If the trainer is None.

        """
        if self.trainer is None:
            raise ValueError(
                "If subset_ckpt_filenames is None, "
                "trainer must be provided to train the models."
            )

        subset_ckpt_filenames = []
        for i in range(self.m):
            subset = self.subsets[i]
            subset_model = self.train_subset_model(
                model=self.model,
                subset=subset,
                trainer=self.trainer,
                batch_size=self.batch_size,
                trainer_fit_kwargs=self.trainer_fit_kwargs,
            )

            ckpt_fname = f"{self.cache_dir}/{self.model_id}_lds_model_{i}.ckpt"
            subset_ckpt_filenames.append(ckpt_fname)
            model_ckpt_path = os.path.join(self.cache_dir, ckpt_fname)
            torch.save(subset_model.state_dict(), model_ckpt_path)

        return subset_ckpt_filenames

    @staticmethod
    def train_subset_model(
        model: Union[torch.nn.Module, L.LightningModule],
        subset: torch.utils.data.Subset,
        trainer: Union[L.Trainer, BaseTrainer],
        batch_size: int = 32,
        trainer_fit_kwargs: Optional[dict] = None,
        reinit: bool = True,
    ):
        """Train a model on a subset of the data.

        Parameters
        ----------
        model : Union[torch.nn.Module, L.LightningModule]
            The model to train.
        subset : torch.utils.data.Subset
            The subset of the dataset to train on.
        trainer : Union[L.Trainer, BaseTrainer]
            The trainer to use for training the model.
        batch_size : int, optional
            Batch size for training, by default 32.
        trainer_fit_kwargs : Optional[dict], optional
            Additional keyword arguments for the trainer's fit method,
            by default None.
        reinit : bool, optional
            If True, reinitialize model weights before training
            (train from scratch). Required for proper LDS evaluation.
            By default True.

        Returns
        -------
        Union[torch.nn.Module, L.LightningModule]
            The trained model.

        """
        subset_model = deepcopy(model)
        if reinit:
            for module in subset_model.modules():
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()
        subset_loader = DataLoader(
            subset, batch_size=batch_size, shuffle=True
        )
        trainer_fit_kwargs = trainer_fit_kwargs or {}
        if isinstance(trainer, L.Trainer):
            if not isinstance(subset_model, L.LightningModule):
                raise ValueError(
                    "Model should be a LightningModule if Trainer is a "
                    "Lightning Trainer"
                )

            trainer.fit(
                model=subset_model,
                train_dataloaders=subset_loader,
                **trainer_fit_kwargs,
            )

        elif isinstance(trainer, BaseTrainer):
            if not isinstance(subset_model, torch.nn.Module):
                raise ValueError(
                    "Model should be a torch.nn.Module if Trainer is a "
                    "BaseTrainer"
                )
            trainer.fit(
                model=subset_model,
                train_dataloaders=subset_loader,
                **trainer_fit_kwargs,
            )
        else:
            raise ValueError(
                "Trainer should be either a Lightning Trainer or "
                "a BaseTrainer."
            )
        return subset_model

    def load_counterfactual_model(self, idx: int):
        """Load a model checkpoint.

        Parameters
        ----------
        idx : int
            Index of the model to load.

        Returns
        -------
        torch.nn.Module
            The loaded model.

        """
        subset_model = deepcopy(self.model)
        self.checkpoints_load_func(
            subset_model, self.subset_ckpt_filenames[idx]
        )

        subset_model.to(self.device)
        return subset_model

    def update(
        self,
        explanations: torch.Tensor,
        test_targets: torch.Tensor,
        test_data: torch.Tensor,
        **kwargs,
    ):
        """Update the evaluation scores based on new data.

        Parameters
        ----------
        explanations : torch.Tensor
            The explanation scores for the test data with shape (test_samples,
            dataset_size).
        test_targets : torch.Tensor
            The target values for the explanations.
        test_data : torch.Tensor
            The test data used for evaluation.
        kwargs: Any
            Additional keyword arguments

        Returns
        -------
        None

        """
        predicted_output_list = []
        model_output_list = []

        for s in range(self.m):
            subset = self.subsets[s]
            subset_indices = subset.indices

            explanation_subset = explanations[:, subset_indices]

            g_tau = explanation_subset.sum(dim=1)

            predicted_output_list.append(g_tau)

            counterfactual_model = self.load_counterfactual_model(s)
            counterfactual_output = counterfactual_model(test_data).detach()
            # We take softmax since we want the rank
            # correlation of probabilities
            # The original definition computes the rank
            # correlation of p/1-p
            # So it is skipped to avoid overflow errors.
            # This operation conserves the ranking of the data
            # We also take logsoftmax
            # to avoid underflow issues at the softmax output
            if (
                counterfactual_output.ndim == 1
                or counterfactual_output.shape[1] == 1
            ):
                counterfactual_output = counterfactual_output.squeeze()
            else:
                counterfactual_output = log_softmax(
                    counterfactual_output, dim=-1
                )
                counterfactual_output = counterfactual_output.gather(
                    1, test_targets.unsqueeze(1)
                ).squeeze(1)

            model_output_list.append(counterfactual_output)

        model_outputs = torch.stack(model_output_list, dim=1)
        predicted_outputs = torch.stack(predicted_output_list, dim=1)

        batch_lds_scores = self.corr_measure(model_outputs, predicted_outputs)

        self.results["scores"].append(batch_lds_scores)

    def reset(self, *args, **kwargs):
        """Reset the LDS score."""
        self.results = {"scores": []}

    def load_state_dict(self, state_dict: dict):
        """Load the state of the metric.

        Parameters
        ----------
        state_dict : dict
            The state dictionary of the metric

        """
        self.results["scores"] = state_dict["scores"]
        self.subsets = state_dict["subsets"]

    def state_dict(self, *args, **kwargs):
        """Return the current state of the metric.

        Returns
        -------
        dict
            The current state of the LDS metric, containing scores and subsets.

        """
        return {
            "scores": self.results["scores"],
            "subsets": self.subsets,
        }

    def compute(self, *args, **kwargs):
        """Compute and return the mean score.

        Returns
        -------
            dict: A dictionary containing the mean score.

        """
        return {"score": torch.cat(self.results["scores"]).mean().item()}
