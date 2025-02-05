"""Kronfluence data attribution wrapper."""

from typing import Any, Callable, List, Optional, Union

import torch
from datasets import DatasetDict  # type: ignore
from kronfluence.analyzer import Analyzer, prepare_model  # type: ignore
from kronfluence.arguments import (  # type: ignore
    FactorArguments,
    ScoreArguments,
)
from kronfluence.task import Task  # type: ignore
from kronfluence.utils.dataset import DataLoaderKwargs  # type: ignore
from torch import nn
from torch.utils.data import Dataset

from quanda.explainers.base import Explainer
from quanda.explainers.utils import (
    explain_fn_from_explainer,
    self_influence_fn_from_explainer,
)
from quanda.utils.common import process_targets
from quanda.utils.tasks import TaskLiterals


class Kronfluence(Explainer):
    """Class for Kronfluence Explainer.

    This explainer uses the Kronfluence package [2] to compute training data
    attributions.

    Notes
    -----
    The user is referred to the Kronfluences' codebase [2] for detailed
    explanations of the parameters.

    References
    ----------
    (1) Roger Grosse, Juhan Bae, Cem Anil, Nelson Elhage, Alex Tamkin,
    Amirhossein Tajdini, Benoit Steiner,
        Dustin Li, Esin Durmus, Ethan Perez, Evan Hubinger, Kamilė Lukošiūtė,
        Karina Nguyen, Nicholas Joseph,
        Sam McCandlish, Jared Kaplan, Samuel R. Bowman. (2023).
        "Studying large language model generalization with influence
        functions". arXiv preprint arXiv:2308.03296.

    (2) https://github.com/pomonam/kronfluence

    """

    accepted_tasks: List[TaskLiterals] = [
        "image_classification",
        "text_classification",
    ]

    def __init__(
        self,
        model: nn.Module,
        task_module: Task,
        train_dataset: Dataset,
        task: TaskLiterals = "image_classification",
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[Callable[..., Any]] = None,
        batch_size: int = 1,
        device: Union[str, torch.device] = "cpu",
        analysis_name: str = "kronfluence_analysis",
        factors_name: str = "initial_factor",
        factor_args: FactorArguments = None,
        scores_name: str = "initial_score",
        score_args: ScoreArguments = None,
        dataloader_kwargs: DataLoaderKwargs = None,
        overwrite_output_dir: bool = True,
    ):
        """Initialize the `Kronfluence` explainer.

        Parameters
        ----------
        model : nn.Module
            The trained model for which influence scores will be computed.
        task_module : kronfluence.task.Task
            The task associated with the model.
        train_dataset : torch.utils.data.Dataset
            Training dataset to be used for the influence computation.
        task: TaskLiterals, optional
            Task type of the model. Defaults to "image_classification".
            Possible options: "image_classification", "text_classification",
            "causal_lm".
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s), defaults to None.
        checkpoints_load_func : Optional[Callable[..., Any]], optional
            Function to load the model from the checkpoint file, takes
            (model, checkpoint path) as two arguments, by default None.
        batch_size : int, optional
            Batch size used for iterating over the dataset. Defaults to 1.
        device : Union[str, torch.device], optional
            Device to run the computation on. Defaults to "cpu".
        analysis_name : str, optional
            Unique identifier for the analysis. Defaults to
            "kronfluence_analysis".
        factors_name : str, optional
            Unique identifier for the factor. Defaults to "initial_factor".
        factor_args : FactorArguments, optional
            Arguments for factor computation. Defaults to None.
        scores_name : str, optional
            The unique identifier for the score. Defaults to "initial_score".
        score_args : ScoreArguments, optional
            Arguments for score computation. Defaults to None.
        dataloader_kwargs : DataLoaderKwargs, optional
            DataLoader arguments. Defaults to None.
        overwrite_output_dir : bool, optional
            Whether to overwrite stored results. Defaults to True.

        """
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            task=task,
            checkpoints=checkpoints,
            checkpoints_load_func=checkpoints_load_func,
        )
        self.batch_size = batch_size
        self.device = device

        self.task = task_module
        self.model = self._prepare_model()

        self.analysis_name = analysis_name
        self.factor_args = factor_args
        self.factors_name = factors_name
        self.scores_name = scores_name
        self.score_args = score_args
        self.overwrite_output_dir = overwrite_output_dir

        self.analyzer = Analyzer(
            analysis_name=self.analysis_name,
            model=self.model,
            task=self.task,
        )

        if dataloader_kwargs:
            self.analyzer.set_dataloader_kwargs(dataloader_kwargs)

        self.analyzer.fit_all_factors(
            factors_name=self.factors_name,
            dataset=self.train_dataset,
            factor_args=self.factor_args,
            overwrite_output_dir=self.overwrite_output_dir,
        )

    def _prepare_model(self) -> nn.Module:
        """Move model to the specified device and calling `prepare_model`.

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
        test_data: Union[torch.Tensor, DatasetDict],
        targets: Union[List[int], torch.Tensor],
        scores_name: Optional[str] = None,
        score_args: ScoreArguments = None,
        overwrite_output_dir: bool = True,
    ) -> torch.Tensor:
        """Compute influence scores for the test samples.

        Parameters
        ----------
        test_data : torch.Tensor
            Test samples for which influence scores are computed.
        targets : Union[List[int], torch.Tensor]
            Labels for the test samples. This argument is required.
        scores_name : str, optional
            The unique identifier for the score. Overrides the instance
            variable if provided.
        score_args : ScoreArguments, optional
            Arguments for score computation. Overrides the instance variable
            if provided.
        overwrite_output_dir : bool, optional
            Whether to overwrite stored results. Defaults to True.

        Returns
        -------
        torch.Tensor
            2D Tensor of shape (test_samples, train_dataset_size) containing
            the influence scores.

        """
        targets = process_targets(targets, self.device)
        test_dataset = torch.utils.data.TensorDataset(test_data, targets)

        scores_name = scores_name or self.scores_name
        score_args = score_args or self.score_args

        self.analyzer.compute_pairwise_scores(
            scores_name=scores_name,
            factors_name=self.factors_name,
            query_dataset=test_dataset,
            train_dataset=self.train_dataset,
            per_device_query_batch_size=self.batch_size,
            score_args=score_args,
            overwrite_output_dir=overwrite_output_dir,
        )
        scores = self.analyzer.load_pairwise_scores(
            scores_name=self.scores_name
        )["all_modules"]

        return scores

    def self_influence(
        self,
        batch_size: int = 1,
        scores_name: Optional[str] = None,
        score_args: ScoreArguments = None,
        overwrite_output_dir: bool = True,
    ) -> torch.Tensor:
        """Compute self-influence scores.

        Parameters
        ----------
        batch_size : int, optional
            Batch size used for iterating over the dataset. This argument is
            ignored.
        scores_name : str, optional
            The unique identifier for the score. Overrides the instance
            variable if provided.
        score_args : ScoreArguments, optional
            Arguments for score computation. Overrides the instance variable
            if provided.
        overwrite_output_dir : bool, optional
            Whether to overwrite stored results. Defaults to True.

        Returns
        -------
        torch.Tensor
            Self-influence scores for each datapoint in train_dataset.

        """
        scores_name = scores_name or self.scores_name
        score_args = score_args or self.score_args

        self.analyzer.compute_self_scores(
            scores_name=scores_name,
            factors_name=self.factors_name,
            train_dataset=self.train_dataset,
            score_args=score_args,
            overwrite_output_dir=overwrite_output_dir,
        )

        scores = self.analyzer.load_self_scores(scores_name=self.scores_name)[
            "all_modules"
        ]

        return scores


def kronfluence_explain(
    model: nn.Module,
    task_module: Task,
    test_tensor: Union[torch.Tensor, DatasetDict],
    explanation_targets: Union[List[int], torch.Tensor],
    train_dataset: Dataset,
    checkpoints: Optional[Union[str, List[str]]] = None,
    checkpoints_load_func: Optional[Callable[..., Any]] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Functional interface for the `Kronfluence` explainer.

    Parameters
    ----------
    model : Union[torch.nn.Module, pl.LightningModule]
        The model to be used for the influence computation.
    task_module : kronfluence.task.Task
        The task associated with the model.
    test_tensor : torch.Tensor
        Test samples for which influence scores are computed.
    explanation_targets : Union[List[int], torch.Tensor]
        Labels for the test samples.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    checkpoints : Optional[Union[str, List[str]]], optional
        Path to the model checkpoint file(s), defaults to None.
    checkpoints_load_func : Optional[Callable[..., Any]], optional
        Function to load the model from the checkpoint file, takes
        (model, checkpoint path) as two arguments, by default None.
    **kwargs : Any
        Additional keyword arguments passed to the explainer.

    Returns
    -------
    torch.Tensor
        2D Tensor of shape (test_samples, train_dataset_size) containing the
        influence scores.

    """
    return explain_fn_from_explainer(
        explainer_cls=Kronfluence,
        model=model,
        task_module=task_module,
        test_data=test_tensor,
        targets=explanation_targets,
        train_dataset=train_dataset,
        checkpoints=checkpoints,
        checkpoints_load_func=checkpoints_load_func,
        **kwargs,
    )


def kronfluence_self_influence(
    model: nn.Module,
    task: Task,
    train_dataset: Dataset,
    checkpoints: Optional[Union[str, List[str]]] = None,
    checkpoints_load_func: Optional[Callable[..., Any]] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Functional interface for `Kronfluence` explainer.

    Parameters
    ----------
    model : Union[torch.nn.Module, pl.LightningModule]
        The model to be used for the influence computation.
    task : kronfluence.task.Task
        The task associated with the model.
    train_dataset : torch.utils.data.Dataset
        Training dataset to be used for the influence computation.
    checkpoints : Optional[Union[str, List[str]]], optional
        Path to the model checkpoint file(s), defaults to None.
    checkpoints_load_func : Optional[Callable[..., Any]], optional
        Function to load the model from the checkpoint file, takes
        (model, checkpoint path) as two arguments, by default None.
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
        task_module=task,
        train_dataset=train_dataset,
        checkpoints=checkpoints,
        checkpoints_load_func=checkpoints_load_func,
        **kwargs,
    )
