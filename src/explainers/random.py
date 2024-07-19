from typing import Any, List, Optional, Union

import torch

from src.explainers.base import BaseExplainer
from src.utils.common import cache_result


class RandomExplainer(BaseExplainer):
    """
    The most basic version of a random explainer.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_id: str,
        cache_dir: Optional[str],
        train_dataset: torch.utils.data.Dataset,
        seed: int = 27,
        device: Union[str, torch.device] = "cpu",
        **kwargs,
    ):
        super().__init__(
            model=model,
            model_id=model_id,
            cache_dir=cache_dir,
            train_dataset=train_dataset,
            device=device,
        )
        self.seed = seed
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(self.seed)

    def explain(self, test: torch.Tensor, targets: Optional[Union[List[int], torch.Tensor]] = None):
        """
        Random explainer does not explain anything, just returns random values.

        TODO: shall the explainer always return the same values for the same test input?

        Parameters
        ----------
        test
        targets

        Returns
        -------

        """
        return torch.rand(test.size(0), self.dataset_length, generator=self.generator, device=self.device)

    @cache_result
    def self_influence(self, batch_size: int = 32, **kwargs: Any) -> torch.Tensor:
        """
        Random self-influence is just a vector of random values of the length of the dataset.

        Parameters
        ----------
        batch_size
        kwargs

        Returns
        -------

        """
        return torch.rand(self.dataset_length, generator=self.generator, device=self.device)
