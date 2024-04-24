from functools import reduce
from typing import Any

def _get_module_from_name(model: torch.nn.Module, layer_name: str) -> Any:
    return reduce(getattr, layer_name.split("."), model)
