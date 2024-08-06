import glob
import os
import warnings
from typing import Any, List, Optional, Tuple, Union

import torch
from captum.attr import LayerActivation  # type: ignore
from torch import Tensor
from torch.utils.data import DataLoader

from quanda.utils.common import _get_module_from_name
from quanda.utils.datasets import ActivationDataset
from quanda.utils.explanations import BatchedCachedExplanations


class Cache:
    """
    Abstract class for caching.
    """

    def __init__(self):
        pass

    @staticmethod
    def save(*args, **kwargs) -> None:
        raise NotImplementedError

    @staticmethod
    def load(*args, **kwargs) -> Any:
        raise NotImplementedError

    @staticmethod
    def exists(*args, **kwargs) -> bool:
        raise NotImplementedError


class TensorCache(Cache):
    def __init__(self):
        super().__init__()

    @staticmethod
    def save(path: str, file_id: str, indices: Tensor) -> None:
        file_path = os.path.join(path, file_id)
        return torch.save(indices, file_path)

    @staticmethod
    def load(path: str, file_id: str, device: str = "cpu") -> Tensor:
        file_path = os.path.join(path, file_id)
        return torch.load(file_path, map_location=device)

    @staticmethod
    def exists(path: str, file_id: str) -> bool:
        file_path = os.path.join(path, file_id)
        return os.path.isfile(file_path)


class ExplanationsCache(Cache):
    def __init__(self):
        super().__init__()

    @staticmethod
    def exists(
        path: str,
        num_id: Optional[Union[str, int]] = None,
    ) -> bool:
        av_filesearch = os.path.join(path, "*.pt" if num_id is None else f"{num_id}.pt")
        return os.path.exists(path) and len(glob.glob(av_filesearch)) > 0

    @staticmethod
    def save(
        path: str,
        exp_tensors: List[Tensor],
        num_id: Union[str, int],
    ) -> None:
        av_save_fl_path = os.path.join(path, f"{num_id}.pt")
        torch.save(exp_tensors, av_save_fl_path)

    @staticmethod
    def load(
        path: str,
        device: str = "cpu",
    ) -> BatchedCachedExplanations:
        if os.path.exists(path):
            xpl_dataset = BatchedCachedExplanations(cache_dir=path, device=device)
            return xpl_dataset
        else:
            raise RuntimeError(f"Activation vectors were not found at path {path}")


class ActivationsCache(Cache):
    """
    Inspired by https://github.com/pytorch/captum/blob/master/captum/_utils/av.py.
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def exists(
        path: str,
        layer: str,
        num_id: Optional[Union[str, int]] = None,
    ) -> bool:
        av_dir = os.path.join(path, layer)
        av_filesearch = os.path.join(av_dir, "*.pt" if num_id is None else f"{num_id}.pt")
        return os.path.exists(av_dir) and len(glob.glob(av_filesearch)) > 0

    @staticmethod
    def save(
        path: str,
        layers: str | List[str],
        act_tensors: List[Tensor],
        labels: Tensor,
        num_id: Union[str, int],
    ) -> None:
        if len(layers) != len(act_tensors):
            raise ValueError("The dimension of `layers` and `act_tensors` must match!")

        for i, layer in enumerate(layers):
            layer_dir = os.path.join(path, layer)

            av_save_fl_path = os.path.join(layer_dir, f"{num_id}.pt")

            if not os.path.exists(layer_dir):
                os.makedirs(layer_dir)
            torch.save((act_tensors[i], labels), av_save_fl_path)

    @staticmethod
    def load(
        path: str,
        layer: str,
        device: str = "cpu",
        **kwargs,
    ) -> ActivationDataset:
        layer_dir = os.path.join(path, layer)

        if os.path.exists(layer_dir):
            av_dataset = ActivationDataset(layer_dir, device)
            return av_dataset
        else:
            raise RuntimeError(f"Activation vectors were not found at path {path}")

    @staticmethod
    def _manage_loading_layers(
        path: str,
        layers: Union[str, List[str]],
        load_from_disk: bool = True,
        num_id: Optional[Union[str, int]] = None,
    ) -> str | List[str]:
        unsaved_layers: List[str] = []

        if load_from_disk:
            for layer in layers:
                if not ActivationsCache.exists(path, layer, num_id):
                    unsaved_layers.append(layer)
        else:
            unsaved_layers = [layers] if isinstance(layers, str) else layers
            warnings.warn(
                "Overwriting activations: load_from_disk is set to False. Removing all "
                f"activations matching specified parameters {{path: {path}, "
                f"layers: {layers}}} "
                "before generating new activations."
            )
            for layer in layers:
                files = glob.glob(os.path.join(path, layer))
                for filename in files:
                    os.remove(filename)

        return unsaved_layers

    @staticmethod
    def _compute_and_save_activations(
        path: str,
        model: torch.nn.Module,
        layers: Union[str, List[str]],
        data: Tuple[Tensor, Tensor],
        num_id: Union[str, int],
        additional_forward_args: Any = None,
        load_from_disk: bool = True,
    ) -> None:
        inputs, labels = data
        unsaved_layers = ActivationsCache._manage_loading_layers(
            path,
            layers,
            load_from_disk,
            num_id,
        )
        layer_modules = [_get_module_from_name(model, layer) for layer in unsaved_layers]
        if len(unsaved_layers) > 0:
            layer_act = LayerActivation(model, layer_modules)  # TODO: replace LayerActivation with generic LayerAttibution
            new_activations = layer_act.attribute.__wrapped__(layer_act, inputs, additional_forward_args)  # type: ignore
            ActivationsCache.save(path, unsaved_layers, new_activations, labels, num_id)

    @staticmethod
    def generate_dataset_activations(
        path: str,
        model: torch.nn.Module,
        layers: List[str],
        dataloader: DataLoader,
        load_from_disk: bool = True,
        return_activations: bool = False,
    ) -> Optional[Union[ActivationDataset, List[ActivationDataset]]]:
        unsaved_layers = ActivationsCache._manage_loading_layers(
            path,
            layers,
            load_from_disk,
        )
        if len(unsaved_layers) > 0:
            for i, data in enumerate(dataloader):
                ActivationsCache._compute_and_save_activations(
                    path,
                    model,
                    layers,
                    data,
                    str(i),
                )

        if not return_activations:
            return None
        return [ActivationsCache.load(path, layer) for layer in layers]
