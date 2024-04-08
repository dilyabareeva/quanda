import json
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import torch


class Metric(ABC):
    def __init__(self, train: torch.utils.data.Dataset, test: torch.utils.data.Dataset):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_result(self, dir: str):
        pass

    def write_result(self, result_dict: dict, dir: str, file_name: str) -> None:
        with open(f"{dir}/{file_name}", "w", encoding="utf-8") as f:
            json.dump(self.to_float(result_dict), f, ensure_ascii=False, indent=4)
        print(result_dict)

    @staticmethod
    def to_float(results: Union[dict, str, torch.Tensor]) -> Union[dict, str, torch.Tensor]:
        if isinstance(results, dict):
            return {key: Metric.to_float(r) for key, r in results.items()}
        elif isinstance(results, str):
            return results
        else:
            return np.array(results).astype(float).tolist()
