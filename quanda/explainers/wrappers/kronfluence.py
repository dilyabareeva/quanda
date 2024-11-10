from typing import Any, List, Union

import torch
from kronfluence.analyzer import Analyzer, prepare_model  # type: ignore
from kronfluence.task import Task  # type: ignore
from torch import nn
from torch.utils.data import Dataset

from quanda.explainers.base import Explainer
from quanda.explainers.utils import (
    explain_fn_from_explainer,
    self_influence_fn_from_explainer,
)


class Kronfluence(Explainer):
    """
    Class for Kronfluence Explainer.

    This explainer uses the Kronfluence package [2] to compute training data attributions.

    Notes
    -----
    The user is referred to the Kronfluences' codebase [2] for detailed explanations of the parameters.

    References
    ----------
    (1) Roger Grosse, Juhan Bae, Cem Anil, Nelson Elhage, Alex Tamkin, Amirhossein Tajdini, Benoit Steiner,
        Dustin Li, Esin Durmus, Ethan Perez, Evan Hubinger, Kamilė Lukošiūtė, Karina Nguyen, Nicholas Joseph,
        Sam McCandlish, Jared Kaplan, Samuel R. Bowman. (2023).
        "Studying large language model generalization with influence functions". arXiv preprint arXiv:2308.03296.

    (2) https://github.com/pomonam/kronfluence
    """

    def __init__(
        self,
        model: nn.Module,
        task: Task,
        train_dataset: Dataset,
        batch_size: int = 1,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Initializer for the `Kronfluence` explainer.

        Parameters
        ----------
        model : nn.Module
            The trained model for which influence scores will be computed.
        task : kronfluence.task.Task
            The task associated with the model.
        train_dataset : torch.utils.data.Dataset
            Training dataset to be used for the influence computation.
        batch_size : int, optional
            Batch size used for iterating over the dataset. Defaults to 1.
        device : Union[str, torch.device], optional
            Device to run the computation on. Defaults to "cpu".
        """
        super().__init__(model=model, train_dataset=train_dataset)
        self.batch_size = batch_size
        self.device = device

        self.task = task
        self.model = self._prepare_model()

        self.factors_name = "initial_factor"
        self.analysis_name = "kronfluence_analysis"

        self.analyzer = Analyzer(
            analysis_name=self.analysis_name,
            model=self.model,
            task=self.task,
        )

        self.analyzer.fit_all_factors(
            factors_name=self.factors_name,
            dataset=self.train_dataset,
        )

    def _prepare_model(self) -> nn.Module:
        """
        Prepare the model by moving it to the specified device and calling Kronfluences' `prepare_model` function.

        Returns
        -------
        nn.Module
            The prepared model.
        """
        self.model.to(self.device)
        prepared_model = prepare_model(model=self.model, task=self.task)
        return prepared_model

    def explain(
        self,
        test_tensor: torch.Tensor,
        targets: Union[List[int], torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute influence scores for the test samples.

        Parameters
        ----------
        test_tensor : torch.Tensor
            Test samples for which influence scores are computed.
        targets : Union[List[int], torch.Tensor]
            Labels for the test samples. This argument is required.

        Returns
        -------
        torch.Tensor
            2D Tensor of shape (test_samples, train_dataset_size) containing the influence scores.
        """
        scores_name = "initial_score"
        if isinstance(targets, list):
            targets = torch.tensor(targets)
        test_dataset = torch.utils.data.TensorDataset(test_tensor, targets)

        self.analyzer.compute_pairwise_scores(
            scores_name=scores_name,
            factors_name=self.factors_name,
            query_dataset=test_dataset,
            train_dataset=self.train_dataset,
            per_device_query_batch_size=1,
        )

        # TODO: Make "all_modules" a parameter
        scores = self.analyzer.load_pairwise_scores(scores_name=scores_name)["all_modules"]

        return scores

    def self_influence(self, batch_size: int = 1) -> torch.Tensor:
        """
        Compute self-influence scores.

        Parameters
        ----------
        batch_size : int, optional
            Batch size used for iterating over the dataset. This argument is ignored.

        Returns
        -------
        torch.Tensor
            Self-influence scores for each datapoint in train_dataset.
        """
        scores_name = "initial_score"

        self.analyzer.compute_self_scores(
            scores_name=scores_name,
            factors_name=self.factors_name,
            train_dataset=self.train_dataset,
        )

        # TODO: Make "all_modules" a parameter
        scores = self.analyzer.load_self_scores(scores_name=scores_name)["all_modules"]

        return scores


def kronfluence_explain(
    model: nn.Module,
    task: Task,
    test_tensor: torch.Tensor,
    explanation_targets: Union[List[int], torch.Tensor],
    train_dataset: Dataset,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Functional interface for the `Kronfluence` explainer.

    Parameters
    ----------
    model : Union[torch.nn.Module, pl.LightningModule]
        The model to be used for the influence computation.
    task : kronfluence.task.Task
        The task associated with the model.
    test_tensor : torch.Tensor
        Test samples for which influence scores are computed.
    explanation_targets : Union[List[int], torch.Tensor]
        Labels for the test samples.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    **kwargs : Any
        Additional keyword arguments passed to the explainer.

    Returns
    -------
    torch.Tensor
        2D Tensor of shape (test_samples, train_dataset_size) containing the influence scores.
    """
    return explain_fn_from_explainer(
        explainer_cls=Kronfluence,
        model=model,
        task=task,
        test_tensor=test_tensor,
        targets=explanation_targets,
        train_dataset=train_dataset,
        **kwargs,
    )


def kronfluence_self_influence(
    model: nn.Module,
    task: Task,
    train_dataset: Dataset,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Functional interface for the self-influence scores using the `Kronfluence` explainer.

    Parameters
    ----------
    model : Union[torch.nn.Module, pl.LightningModule]
        The model to be used for the influence computation.
    task : kronfluence.task.Task
        The task associated with the model.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    **kwargs : Any
        Additional keyword arguments passed to the explainer.

    Returns
    -------
    torch.Tensor
        Self-influence scores for each datapoint in train_dataset.
    """
    return self_influence_fn_from_explainer(
        explainer_cls=Kronfluence,
        model=model,
        task=task,
        train_dataset=train_dataset,
        **kwargs,
    )
