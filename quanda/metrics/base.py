"""Base class for metrics."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import datasets  # type: ignore
import torch

from quanda.utils.common import (
    CheckpointLoadFunc,
    get_load_state_dict_func,
    load_last_checkpoint,
)


class Metric(ABC):
    """Base class for metrics."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset: Union[torch.utils.data.Dataset, datasets.Dataset],
        checkpoints: Optional[Union[str, List[str]]] = None,
        checkpoints_load_func: Optional[CheckpointLoadFunc] = None,
    ):
        """Initialize metric.

        Parameters
        ----------
        model : Union[torch.nn.Module, pl.LightningModule]
            The model to be used for the influence computation.
        train_dataset : torch.utils.data.Dataset
            Training dataset to be used for the influence computation.
        checkpoints : Optional[Union[str, List[str]]], optional
            Path to the model checkpoint file(s), defaults to None.
        checkpoints_load_func : Optional[CheckpointLoadFunc], optional
            Function to load the model from the checkpoint file, takes
            (model, checkpoint path) as two arguments, by default None.

        """
        self.device: str
        self.model: torch.nn.Module = model

        def _model_device() -> str:
            first_param = next(model.parameters(), None)
            return "cpu" if first_param is None else str(first_param.device)

        self.device = _model_device()

        if checkpoints_load_func is None:
            self.checkpoints_load_func = get_load_state_dict_func(self.device)
        else:
            self.checkpoints_load_func = checkpoints_load_func

        if checkpoints is None:
            self.checkpoints = []
        else:
            self.checkpoints = (
                checkpoints if isinstance(checkpoints, List) else [checkpoints]
            )
            self.load_last_checkpoint()
            # Checkpoint loaders can move the model to a new device
            # (e.g. `model.to("cuda")`), so re-read after loading.
            self.device = _model_device()
        self.train_dataset: Union[
            torch.utils.data.Dataset, datasets.Dataset
        ] = train_dataset

    @abstractmethod
    def update(
        self,
        explanations: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ):
        """Update the metric with new data.

        Parameters
        ----------
        explanations : torch.Tensor
            Explanations of the test samples.
        *args : Any
            Additional positional arguments.
        **kwargs : Any
            Additional keyword arguments.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    @abstractmethod
    def compute(self) -> Dict[str, Any]:
        """Compute the metric score.

        Returns
        -------
        Dict[str, Any]
            Dictionary keyed by metric output names. All quanda metrics
            return a ``"score"`` entry; some return additional fields.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    def compute_bootstrapped(
        self,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        seed: Optional[int] = None,
    ) -> dict:
        """Bootstrap a test-sample CI around the metric score.

        Parameters
        ----------
        n_bootstrap : int, optional
            Number of bootstrap replicates, by default 1000.
        ci : float, optional
            Confidence level, e.g. 0.95 for a 95% CI, by default 0.95.
        seed : Optional[int], optional
            Seed for the resampling RNG, by default None.

        Returns
        -------
        dict
            Dictionary with ``score``, ``ci_low``, ``ci_high``, ``ci``
            and ``n_bootstrap``.

        """
        scores = self._per_sample_scores()
        if scores is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not support bootstrapping."
            )
        scores = scores.detach().to("cpu").float()
        n = scores.numel()
        if n == 0:
            raise ValueError("No scores accumulated; call update() first.")

        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(seed)
        idx = torch.randint(0, n, (n_bootstrap, n), generator=gen)
        means = scores[idx].mean(dim=1)
        alpha = (1 - ci) / 2
        return {
            "score": scores.mean().item(),
            "ci_low": torch.quantile(means, alpha).item(),
            "ci_high": torch.quantile(means, 1 - alpha).item(),
            "ci": ci,
            "n_bootstrap": n_bootstrap,
        }

    def _per_sample_scores(self) -> Optional[torch.Tensor]:
        """Return accumulated per-sample scores for bootstrapping.

        Returns
        -------
        Optional[torch.Tensor]
            1-D tensor of per-sample scores, or ``None``.

        """
        return None

    @abstractmethod
    def reset(self):
        """Reset the metric state."""
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state_dict: dict):
        """Load the metric state.

        Parameters
        ----------
        state_dict: dict
            The metric state dictionary.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> dict:
        """Get the metric state.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError

    def load_last_checkpoint(self):
        """Load the model from the checkpoint file.

        Parameters
        ----------
        checkpoint : str
            Path to the checkpoint file.

        """
        load_last_checkpoint(
            model=self.model,
            checkpoints=self.checkpoints,
            checkpoints_load_func=self.checkpoints_load_func,
        )
