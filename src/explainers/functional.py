from typing import Dict, List, Optional, Protocol, Union

import torch


class ExplainFunc(Protocol):
    def __call__(
        self,
        model: torch.nn.Module,
        model_id: str,
        cache_dir: Optional[str],
        test_tensor: torch.Tensor,
        train_dataset: torch.utils.data.Dataset,
        device: Union[str, torch.device],
        explanation_targets: Optional[Union[List[int], torch.Tensor]] = None,
        init_kwargs: Optional[Dict] = None,
        explain_kwargs: Optional[Dict] = None,
    ) -> torch.Tensor:
        pass
