from typing import Dict, List, Optional, Protocol, Union

import torch


class ExplainFunc(Protocol):
    def __call__(
        self,
        model: torch.nn.Module,
        model_id: str,
        cache_dir: Optional[str],
        test_tensor: torch.Tensor,
        explanation_targets: Optional[Union[List[int], torch.Tensor]],
        train_dataset: torch.utils.data.Dataset,
        explain_kwargs: Dict,
        init_kwargs: Dict,
        device: Union[str, torch.device],
    ) -> torch.Tensor:
        pass
