import os
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Union

import lightning as L
import torch
from torch.utils.data import DataLoader

from quanda.metrics.base import Metric
from quanda.utils.common import get_load_state_dict_func
from quanda.utils.functions import CorrelationFnLiterals, correlation_functions
from quanda.utils.training import BaseTrainer


class LinearDatamodelingMetric(Metric):
    """
    Metric to evaluate data attribution methods using the Linear Datamodeling Score (LDS).

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

    def __init__(
        self,
        model: Union[torch.nn.Module, L.LightningModule],
        train_dataset: torch.utils.data.Dataset,
        trainer: Union[L.Trainer, BaseTrainer],
        alpha: float = 0.5,
        m: int = 100,
        correlation_fn: Union[Callable, CorrelationFnLiterals] = "spearman",
        trainer_fit_kwargs: Optional[dict] = None,
        load_state_dict: Optional[Callable] = None,
        seed: int = 42,
        batch_size: int = 32,
        subset_ids: Optional[List[List[int]]] = None,
        pretrained_models: Optional[List[torch.nn.Module]] = None,
        model_id: Optional[str] = "0",
        cache_dir: str = "./cache",
    ):
        """
        Initialize the LinearDatamodelingMetric.

        Parameters
        ----------
        model : Union[torch.nn.Module, L.LightningModule]
            The model used to generate attributions.
        train_dataset : torch.utils.data.Dataset
            The training dataset used to train models.
        trainer : Union[L.Trainer, BaseTrainer]
            Trainer object used to fit the model on the sampled subsets.
        alpha : float, optional
            The fraction of the training data to include in each subset, by default 0.5.
        m : int, optional
            Number of subsets to sample, by default 100.
        correlation_fn : Union[Callable, CorrelationFnLiterals], optional
            Correlation function to use, by default "spearman". Can be "spearman", "kendall", or a callable.
        trainer_fit_kwargs : Optional[dict], optional
            Additional keyword arguments for the trainer, by default None.
        load_state_dict : Optional[Callable], optional
            Custom function to load a model state dictionary, by default None.
        seed : Optional[int], optional
            Random seed for reproducibility, by default 42.
        batch_size : int, optional
            Batch size for training, by default 32.
        subset_ids : Optional[List[List[int]]], optional
            A list of pre-defined subset indices, by default None.
        pretrained_models : Optional[List[torch.nn.Module]], optional
            A list of pre-trained models for each subset, by default None.
        model_id : str
            An identifier for the model, by default "0".
        cache_dir : str
            The cache directory, by default "./cache".
        """
        super().__init__(model=model, train_dataset=train_dataset)
        self.device = torch.device("cpu")
        self.cache_dir = cache_dir
        self.model_id = model_id
        self.results: Dict[str, List[torch.Tensor]] = {"scores": []}
        self.m = m
        self.alpha = alpha
        self.trainer = trainer
        self.trainer_fit_kwargs = trainer_fit_kwargs
        self.seed = seed
        self.batch_size = batch_size
        self.subset_ids = subset_ids
        self.pretrained_models = pretrained_models

        if load_state_dict is None:
            self.load_model_state_dict = get_load_state_dict_func(self.device)
        else:
            self.load_model_state_dict = load_state_dict

        # TODO: create a validation utility function
        if isinstance(correlation_fn, str) and correlation_fn in correlation_functions:
            self.corr_measure = correlation_functions[correlation_fn]
        elif callable(correlation_fn):
            self.corr_measure = correlation_fn
        else:
            raise ValueError(
                f"Invalid correlation function: expected one of {list(correlation_functions.keys())} or"
                f"a Callable, but got {self.corr_measure}."
            )

        self.generator = None
        self.generator = torch.Generator() if seed is not None else None
        if self.generator:
            self.generator.manual_seed(self.seed)

        self.subsets = self.sample_subsets(train_dataset)
        self.create_counterfactual_models()

    def sample_subsets(self, dataset):
        """
        Randomly sample m subsets of the training set, each of size alpha * N.

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            The dataset to sample subsets from.

        Returns
        -------
        List[torch.utils.data.Subset]
            A list of m subsets of the training data.
        """
        if self.subset_ids:
            return [torch.utils.data.Subset(dataset, indices) for indices in self.subset_ids]

        N = len(dataset)
        subset_size = int(self.alpha * N)

        subsets = []
        for _ in range(self.m):
            indices = torch.randperm(N, generator=self.generator)[:subset_size].tolist()
            subsets.append(torch.utils.data.Subset(dataset, indices))

        return subsets

    def create_counterfactual_models(self):
        """
        For each subset of the training data, this function creates a new model, trains it
        on the subset and stores it in the cache directory.

        Raises
        ------
        ValueError
            If the model is not a LightningModule and the trainer is a Lightning Trainer.
        ValueError
            If the model is not a torch.nn.Module and the trainer is a BaseTrainer.
        """
        if self.pretrained_models:
            for i, model in enumerate(self.pretrained_models):
                model_ckpt_path = os.path.join(self.cache_dir, f"{self.model_id}_model_{i}.ckpt")
                torch.save(model.state_dict(), model_ckpt_path)
        else:
            for i, subset in enumerate(self.subsets):
                counterfactual_model = deepcopy(self.model)
                subset_loader = DataLoader(subset, batch_size=self.batch_size, shuffle=False)
                self.trainer_fit_kwargs = self.trainer_fit_kwargs or {}
                if isinstance(self.trainer, L.Trainer):
                    if not isinstance(self.model, L.LightningModule):
                        raise ValueError("Model should be a LightningModule if Trainer is a Lightning Trainer")

                    self.trainer.fit(
                        model=self.model,
                        train_dataloaders=subset_loader,
                        **self.trainer_fit_kwargs,
                    )

                elif isinstance(self.trainer, BaseTrainer):
                    if not isinstance(self.model, torch.nn.Module):
                        raise ValueError("Model should be a torch.nn.Module if Trainer is a BaseTrainer")
                    self.trainer.fit(model=counterfactual_model, train_dataloaders=subset_loader, **self.trainer_fit_kwargs)

                model_ckpt_path = os.path.join(self.cache_dir, f"{self.model_id}_model_{i}.ckpt")
                torch.save(counterfactual_model.state_dict(), model_ckpt_path)

    def load_counterfactual_model(self, model_idx: int):
        """
        Load a model checkpoint.

        Parameters
        ----------
        model_idx : int
            Index of the model to load.

        Returns
        -------
        torch.nn.Module
            The loaded model.
        """
        model_ckpt_path = os.path.join(self.cache_dir, f"{self.model_id}_model_{model_idx}.ckpt")
        counterfactual_model = deepcopy(self.model)
        self.load_model_state_dict(counterfactual_model, model_ckpt_path)
        # counterfactual_model.load_state_dict(torch.load(model_ckpt_path, map_location=self.device))
        counterfactual_model.to(self.device)
        return counterfactual_model

    def update(
        self,
        test_tensor: torch.Tensor,
        explanations: torch.Tensor,
        explanation_targets: torch.Tensor,
        **kwargs,
    ):
        """
        Update the evaluation scores based on the provided test data and explanations.

        Parameters
        ----------
        test_tensor : torch.Tensor
            The test data used for evaluation.
        explanations : torch.Tensor
            The explanation scores for the test data with shape (test_samples, dataset_size).
        explanation_targets : torch.Tensor
            The target values for the explanations.

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
            counterfactual_output = counterfactual_model(test_tensor).detach()

            if counterfactual_output.ndim == 1 or counterfactual_output.shape[1] == 1:
                counterfactual_output = counterfactual_output.squeeze()
            else:
                counterfactual_output = counterfactual_output.gather(1, explanation_targets.unsqueeze(1)).squeeze(1)

            model_output_list.append(counterfactual_output)

        model_outputs = torch.stack(model_output_list, dim=1)
        predicted_outputs = torch.stack(predicted_output_list, dim=1)

        batch_lds_scores = self.corr_measure(model_outputs, predicted_outputs)

        self.results["scores"].append(batch_lds_scores)

    def reset(self, *args, **kwargs):
        """
        Reset the LDS score and resample subsets of the training data.
        """
        self.results = {"scores": []}
        self.subsets = self.sample_subsets(dataset=self.train_dataset)

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        """
        Load the state of the metric.

        Parameters
        ----------
        state_dict : dict
            The state dictionary of the metric
        """
        self.results["scores"] = state_dict["scores"]
        self.subsets = state_dict["subsets"]

    def state_dict(self, *args, **kwargs):
        """
        Return the current state of the metric.

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
        """
        Compute and return the mean score.

        Returns
        -------
            dict: A dictionary containing the mean score.
        """
        return {"score": torch.cat(self.results["scores"]).mean().item()}
