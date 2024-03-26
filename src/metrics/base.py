from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch


class Metric(ABC):
    name = "BaseMetricClass"

    @abstractmethod
    def __init__(self, train: torch.utils.data.Dataset, test: torch.utils.data.Dataset):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_result(self, dir: str):
        pass

    @staticmethod
    def to_float(results: Union[dict, str, torch.Tensor]) -> Union[dict, str, torch.Tensor]:
        if isinstance(results, dict):
            return {key: Metric.to_float(r) for key, r in results.items()}
        elif isinstance(results, str):
            return results
        else:
            return np.array(results).astype(float).tolist()
