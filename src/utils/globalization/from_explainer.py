from typing import Callable, Optional

import torch

from src.utils.common import make_func
from src.utils.globalization.base import Globalization


class GlobalizationFromSingleImageAttributor(Globalization):
    def __init__(
        self,
        training_dataset: torch.utils.data.Dataset,
        model: torch.nn.Module,
        attributor_fn: Callable,
        attributor_fn_kwargs: Optional[dict] = None,
    ):
        # why is it called attributor
        super().__init__(training_dataset=training_dataset)
        self.attributor_fn = make_func(func=attributor_fn, func_kwargs=attributor_fn_kwargs, model=self.model)
        self.model = model

    def compute_self_influences(self):
        for i, (x, _) in enumerate(self.training_dataset):
            self.scores[i] = self.attributor_fn(datapoint=x)

    def update_self_influences(self, self_influences):
        self.scores = self_influences
