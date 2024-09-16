from typing import Any, List, Optional, Protocol, Union

import torch


class ExplainFunc(Protocol):
    """Protocol for functional explainers."""

    def __call__(
        self,
        model: torch.nn.Module,
        model_id: str,
        test_tensor: torch.Tensor,
        train_dataset: torch.utils.data.Dataset,
        device: Union[str, torch.device],
        cache_dir: str = "./cache",
        explanation_targets: Optional[Union[List[int], torch.Tensor]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        pass


class ExplainFuncMini(Protocol):
    """Explain function typing."""

    def __call__(
        self,
        test_tensor: torch.Tensor,
        explanation_targets: Optional[Union[List[int], torch.Tensor]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        pass
