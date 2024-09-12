from typing import Any, List, Optional, Union

import torch

from quanda.explainers import Explainer
from quanda.utils.common import cache_result


class RandomExplainer(Explainer):
    """
    The most basic version of a random explainer.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_id: str,
        train_dataset: torch.utils.data.Dataset,
        cache_dir: str = "./cache",
        seed: int = 27,
        **kwargs,
    ):
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
