"""Model randomization metric."""

import copy
import inspect
import os
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch

from quanda.metrics.base import Metric
from quanda.utils.common import (
    CheckpointLoadFunc,
    get_parent_module_from_name,
    move_ds_item_to_device,
)
from quanda.utils.functions import CorrelationFnLiterals, correlation_functions


class ModelRandomizationMetric(Metric):
    """Evaluate the dependence of the attributions on the model parameters.

    References
    ----------
    1) Hanawa, K., Yokoi, S., Hara, S., & Inui, K. (2021). Evaluation of
    similarity-based explanations. In International Conference on Learning
    Representations.

    2) Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., & Kim,
    B. (2018). Sanity checks for saliency maps. In Advances in Neural
    Information Processing Systems (Vol. 31).

    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_id: str,
        cache_dir: str,
        train_dataset: torch.utils.data.Dataset,
        checkpoints: Union[str, List[str]],
        explainer_cls: Type,
        expl_kwargs: Optional[dict] = None,
        checkpoints_load_func: Optional[CheckpointLoadFunc] = None,
        correlation_fn: Union[Callable, CorrelationFnLiterals] = "spearman",
        n_rand_models: int = 1,
        seed: int = 42,
    ):
        """Initialize the ModelRandomizationMetric.

        Parameters
        ----------
        model : torch.nn.Module
            The model used to generate attributions.
        model_id : str
            The identifier of the model.
        cache_dir : str
            The cache directory.
        train_dataset : torch.utils.data.Dataset
            The training dataset used to train `model`.
        checkpoints : Union[str, List[str]]
            Path to the model checkpoint file(s). Required because
            model randomization needs to load and randomize
            checkpoints.
        explainer_cls : Type
            The class of the explainer to evaluate.
        expl_kwargs : Optional[dict], optional
            Additional keyword arguments for the explainer,
            by default None.
        checkpoints_load_func : Optional[CheckpointLoadFunc], optional
            Function to load the model from the checkpoint file,
            takes (model, checkpoint path) as two arguments,
            by default None.
        correlation_fn : Union[Callable, CorrelationFnLiterals],
            optional. The correlation function to use,
            by default "spearman".
            Can be "spearman", "kendall" or a callable.
        n_rand_models : int, optional
            Number of independently randomized models to average the
            metric over, by default 1. Each model uses a distinct random
            seed; ``compute`` returns the mean and standard deviation of
            the per-model scores.
        seed : int, optional
            The random seed, by default 42.

        """
        super().__init__(
            model=model,
            checkpoints=checkpoints,
            train_dataset=train_dataset,
            checkpoints_load_func=checkpoints_load_func,
        )

        self.expl_kwargs = copy.copy(expl_kwargs) if expl_kwargs else {}
        self.model_id = model_id
        self.cache_dir = cache_dir
        # create cache directory if it does not exist
        os.makedirs(self.cache_dir, exist_ok=True)

        self.seed = seed
        self.n_rand_models = n_rand_models

        explainer_params = inspect.signature(explainer_cls.__init__).parameters

        # Build one randomized model + explainer per requested random model.
        self.rand_models: List[torch.nn.Module] = []
        self.rand_checkpoints: List[List[str]] = []
        self.rand_explainers: List = []
        resolved_expl_kwargs: List[dict] = []
        for model_idx in range(self.n_rand_models):
            rand_model, rand_checkpoint = self._randomize_model(model_idx)
            self.rand_models.append(rand_model)
            self.rand_checkpoints.append(rand_checkpoint)

            expl_kwargs = copy.copy(self.expl_kwargs)
            if "model_id" in explainer_params:
                base_id = self.expl_kwargs.get("model_id", self.model_id)
                expl_kwargs["model_id"] = f"{base_id}_rand_{model_idx}"

            # this is needed for the random explainer: otherwise the
            # correlation is 1.0
            if "seed" in explainer_params:
                expl_kwargs["seed"] = (
                    self.expl_kwargs.get("seed", self.seed) + 1 + model_idx
                )

            # Never reuse cached artifacts for randomized models.
            if "load_from_disk" in explainer_params:
                expl_kwargs["load_from_disk"] = False

            resolved_expl_kwargs.append(expl_kwargs)
            self.rand_explainers.append(
                explainer_cls(
                    model=rand_model,
                    checkpoints=rand_checkpoint,
                    train_dataset=train_dataset,
                    **expl_kwargs,
                )
            )

        # Backward-compatible handles to the first randomized model.
        self.expl_kwargs = resolved_expl_kwargs[0]
        self.rand_model = self.rand_models[0]
        self.rand_checkpoint = self.rand_checkpoints[0]
        self.rand_explainer = self.rand_explainers[0]

        self.results: Dict[str, List] = {
            "scores": [[] for _ in range(self.n_rand_models)]
        }

        # TODO: create a validation utility function
        if (
            isinstance(correlation_fn, str)
            and correlation_fn in correlation_functions
        ):
            self.corr_measure = correlation_functions[correlation_fn]
        elif callable(correlation_fn):
            self.corr_measure = correlation_fn
        else:
            raise ValueError(
                f"Invalid correlation function: expected one of "
                f"{list(correlation_functions.keys())} or"
                f"a Callable, but got {self.corr_measure}."
            )

    def update(
        self,
        explanations: torch.Tensor,
        test_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
        test_targets: Optional[torch.Tensor] = None,
    ):
        """Update the evaluation scores based on the provided data.

        Parameters
        ----------
        explanations : torch.Tensor
            The explanations generated by the model.
        test_data : Union[torch.Tensor, Dict[str, torch.Tensor]]
            The test data used for evaluation.
        test_targets : Optional[torch.Tensor], optional
            The target values for the explanations, by default None.

        Raises
        ------
        ValueError
            If the original or the randomized explanations contain
            non-finite values.

        """
        explanations = explanations.to(self.device)
        test_data = move_ds_item_to_device(test_data, self.device)
        if test_targets is not None:
            test_targets = test_targets.to(self.device)

        if not torch.isfinite(explanations).all():
            raise ValueError(
                "The explanations of the original model contain non-finite "
                "values; the rank correlation would be meaningless."
            )

        for model_idx, rand_explainer in enumerate(self.rand_explainers):
            rand_explanations = rand_explainer.explain(
                test_data=test_data, targets=test_targets
            ).to(self.device)

            if not torch.isfinite(rand_explanations).all():
                raise ValueError(
                    f"Randomized model {model_idx} produced non-finite "
                    "explanations; the rank correlation would be "
                    "meaningless."
                )

            corrs = self.corr_measure(explanations, rand_explanations)
            self.results["scores"][model_idx].append(corrs)

    def compute(self):
        """Compute and return the mean and std score across random models.

        Returns
        -------
            dict: A dictionary with ``"score"`` and ``"mean"`` (both equal to
            the mean per-model correlation), ``"std"`` (the standard
            deviation of the per-model correlations, ``0.0`` for a single
            random model) and ``"per_model_scores"`` (the individual
            correlation of each randomized model, in model order).

        """
        per_model_means = torch.stack(
            [
                torch.cat(model_scores).mean()
                for model_scores in self.results["scores"]
            ]
        )
        mean = per_model_means.mean().item()
        std = (
            per_model_means.std(unbiased=False).item()
            if self.n_rand_models > 1
            else 0.0
        )
        return {
            "score": mean,
            "mean": mean,
            "std": std,
            "per_model_scores": per_model_means.tolist(),
        }

    def _per_sample_scores(self) -> Optional[torch.Tensor]:
        """Return per-sample correlations against the randomized models."""
        all_scores = [
            torch.cat(model_scores)
            for model_scores in self.results["scores"]
            if model_scores
        ]
        if not all_scores:
            return torch.empty(0)
        return torch.cat(all_scores)

    def reset(self):
        """Reset the state of the model randomization.

        This method resets the state of the model randomization by clearing the
        results and re-randomizing the models using the `_randomize_model`
        method.

        """
        self.results = {"scores": [[] for _ in range(self.n_rand_models)]}
        self.rand_models = []
        self.rand_checkpoints = []
        for model_idx in range(self.n_rand_models):
            rand_model, rand_checkpoint = self._randomize_model(model_idx)
            self.rand_models.append(rand_model)
            self.rand_checkpoints.append(rand_checkpoint)
        self.rand_model = self.rand_models[0]
        self.rand_checkpoint = self.rand_checkpoints[0]

    def state_dict(self) -> Dict:
        """Return the state of the metric.

        Returns
        -------
        Dict
            The state of the metric

        """
        state_dict = {
            "results_dict": self.results,
            "rnd_models": [m.state_dict() for m in self.rand_models],
        }
        return state_dict

    def load_state_dict(self, state_dict: dict):
        """Load the state of the metric.

        Parameters
        ----------
        state_dict : dict
            The state dictionary of the metric

        """
        self.results = state_dict["results_dict"]
        for rand_model, model_state in zip(
            self.rand_models, state_dict["rnd_models"]
        ):
            rand_model.load_state_dict(model_state)

    def _randomize_parameter(self, param, parent, param_name, generator, seed):
        """Reset or randomize a parameter.

        Parameters
        ----------
        param : torch.Tensor
            The parameter tensor.
        parent : torch.nn.Module
            The parent module of the parameter.
        param_name : str
            The name of the parameter.
        generator : torch.Generator
            The random generator used to draw new parameter values.
        seed : int
            The seed used to reset parameters of modules exposing
            ``reset_parameters``.

        """
        if hasattr(parent, "reset_parameters"):
            torch.manual_seed(seed)
            parent.reset_parameters()
        else:
            torch.nn.init.normal_(param, generator=generator)
            parent.__setattr__(param_name, torch.nn.Parameter(param))

    def _randomize_model(
        self, model_idx: int = 0
    ) -> Tuple[torch.nn.Module, List[str]]:
        """Randomize the model parameters.

        Parameters
        ----------
        model_idx : int, optional
            Index of the random model being built, by default 0. It offsets
            the random seed so that each of the ``n_rand_models`` models is
            randomized independently.

        Returns
        -------
        torch.nn.Module
            The randomized model.

        """
        seed = self.seed + model_idx
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        rand_model = copy.deepcopy(self.model)
        rand_checkpoints = []

        for i, chckpt in enumerate(self.checkpoints):
            self.checkpoints_load_func(rand_model, chckpt)

            for name, param in list(rand_model.named_parameters()):
                parent = get_parent_module_from_name(rand_model, name)
                param_name = name.split(".")[-1]
                self._randomize_parameter(
                    param, parent, param_name, generator, seed
                )

            # Save randomized checkpoint
            chckpt_path = os.path.join(
                self.cache_dir, f"{self.model_id}_rand_{model_idx}_{i}.pth"
            )
            torch.save(rand_model.state_dict(), chckpt_path)
            rand_checkpoints.append(chckpt_path)

        return rand_model, rand_checkpoints
