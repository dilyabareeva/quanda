from typing import Any, List, Optional, Union

import torch

from quanda.explainers import Explainer
from quanda.utils.common import cache_result


class RandomExplainer(Explainer):
    """
    The most basic version of a random explainer.

    Attributes
    ----------
    model : Union[torch.nn.Module, pl.LightningModule]
        Trained model to be explained.
    cache_dir : str
        Directory to be used for caching.
    train_dataset : torch.utils.data.Dataset
        Training dataset that was used to train the model.
    model_id : Optional[str], optional
        An identifier for the model. This field is generally not used and is included for completeness, defaults to None.

    Methods
    -------
    explain(test: torch.Tensor, targets: Optional[Union[List[int], torch.Tensor]] = None) -> torch.Tensor
        Random explainer does not explain anything, just returns random values.

    self_influence(batch_size: int = 32, **kwargs: Any) -> torch.Tensor
        Random self-influence is just a vector of random values of the length of the training dataset.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_id: str,
        cache_dir: str,
        train_dataset: torch.utils.data.Dataset,
        seed: int = 27,
        **kwargs,
    ):
        """Initializer for RandomExplainer.

        Parameters
        ----------
        model : Union[torch.nn.Module, pl.LightningModule]
            Trained model to be explained.
        cache_dir : str
            Directory to be used for caching.
        train_dataset : torch.utils.data.Dataset
            Training dataset that was used to train the model.
        model_id : Optional[str], optional
            An identifier for the model. This field is generally not used and is included for completeness, defaults to None.

        """
        super().__init__(
            model=model,
            model_id=model_id,
            cache_dir=cache_dir,
            train_dataset=train_dataset,
        )
        self.seed = seed
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

    def explain(self, test: torch.Tensor, targets: Optional[Union[List[int], torch.Tensor]] = None):
        """
        Random explainer does not explain anything, just returns random values.

        Parameters
        ----------
        test : torch.Tensor
            Test points for the model decisions to be explained. Is not used for the `RandomExplainer`.
        targets : Optional[Union[List[int], torch.Tensor]] = None
            The model outputs to be explained.
            Some methods do not need this. Defaults to None. Is not used in `RandomExplainer`.
        Returns
        -------
        torch.Tensor
            Random tensor of shape `(test.shape[0],train_dataset_length)`

        """
        return torch.rand(test.size(0), self.dataset_length, generator=self.generator, device=self.device)

    @cache_result
    def self_influence(self, batch_size: int = 32, **kwargs: Any) -> torch.Tensor:
        """
        Random self-influence is just a vector of random values of the length of the training dataset.

        Parameters
        ----------
        batch_size : int = 32
            `RandomExplainer` does not use this.
        kwargs : Any
            `RandomExplainer` does not use this. TODO:Galip Is this not supposed to be eradicated?

        Returns
        -------
        torch.Tensor
            Random tensor of shape `(train dataset length,)`
        """
        return torch.rand(self.dataset_length, generator=self.generator, device=self.device)
