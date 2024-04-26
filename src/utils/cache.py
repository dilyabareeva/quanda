import glob
import os
import warnings
from typing import Any, List, Optional, Tuple, Union

import torch
from captum.attr import LayerActivation
from torch import Tensor
from torch.utils.data import DataLoader

from utils.common import _get_module_from_name
from utils.datasets.activation_dataset import ActivationDataset


class Cache:
    """
    Abstract class for caching.
    """

    def __init__(self):
        pass

    @staticmethod
    def save(**kwargs) -> None:
        raise NotImplementedError

    @staticmethod
    def load(**kwargs) -> Any:
        raise NotImplementedError

    @staticmethod
    def exists(**kwargs) -> bool:
        raise NotImplementedError


class IndicesCache(Cache):
    def __init__(self):
        super().__init__()

    @staticmethod
    def save(path, file_id, indices) -> None:
        file_path = os.path.join(path, file_id)
        return torch.save(indices, file_path)

    @staticmethod
    def load(path, file_id, device="cpu") -> Tensor:
        file_path = os.path.join(path, file_id)
        return torch.load(file_path, map_location=device)

    @staticmethod
    def exists(path, file_id) -> bool:
        file_path = os.path.join(path, file_id)
        return os.path.isfile(file_path)
    

class ActivationsCache(Cache):
    """
    Inspired by https://github.com/pytorch/captum/blob/master/captum/_utils/av.py.
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def exists(
        path: str,
        layer: Optional[str] = None,
        num_id: Optional[Union[str, int]] = None,
    ) -> bool:
        av_dir = os.path.join(path, layer)
        av_filesearch = os.path.join(av_dir, "*.pt" if num_id is None else f"{num_id}.pt")
        return os.path.exists(av_dir) and len(glob.glob(av_filesearch)) > 0

    @staticmethod
    def save(
        path: str,
        layers: List[str],
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
        layer: Optional[str] = None,
        device: str = "cpu",
    ) -> ActivationDataset:
        layer_dir = os.path.join(path, layer)

        if not os.path.exists(layer_dir):
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
    ) -> List[str]:
        unsaved_layers = []

        if load_from_disk:
            for layer in layers:
                if not ActivationsCache.exists(path, layer, num_id):
                    unsaved_layers.append(layer)
        else:
            unsaved_layers = layers
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
            layer_act = LayerActivation(
                model, layer_modules
            )  # TODO: replace LayerActivation with generic LayerAttibution
            new_activations = layer_act.attribute.__wrapped__(  # type: ignore
                layer_act, inputs, additional_forward_args
            )
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
