from typing import Callable, List, Literal, Optional

import torch
from PIL import Image

from quanda.utils.datasets.transformed import TransformedDataset

SampleFnLiterals = Literal["yellow_box_3c", "box_1c"]


def yellow_box_3c_sample_fn(img: Image.Image) -> Image.Image:
    """
    This function will add a yellow box to the image
    """
    square_size = (15, 15)  # Size of the square
    yellow_square = Image.new("RGB", square_size, (255, 255, 0))  # Create a yellow square
    img.paste(yellow_square, (10, 10))  # Paste it onto the image at the specified position
    return img


def box_1c_sample_fn(img: torch.Tensor) -> torch.Tensor:
    """
    This function will add a box to the single channel image
    """
    square_size = (5, 5)  # Size of the square
    img[:, 8 : 8 + square_size[0], 10 : 10 + square_size[1]] = torch.ones(1, *square_size)
    return img


def get_sample_fn(name: SampleFnLiterals) -> Callable:
    func_dict = {
        "yellow_box_3c": yellow_box_3c_sample_fn,
        "box_1c": box_1c_sample_fn,
    }
    if name in func_dict.keys():
        return func_dict[name]  # type: ignore
    else:
        raise ValueError(f"Unknown sample function {name}")


class SampleTransformationDataset(TransformedDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        n_classes: int,
        sample_fn: Callable,
        dataset_transform: Optional[Callable] = None,
        transform_indices: Optional[List[int]] = None,
        cls_idx: Optional[int] = None,
        p: float = 1.0,
        seed: int = 42,
    ):
        super().__init__(
            dataset=dataset,
            n_classes=n_classes,
            dataset_transform=dataset_transform,
            transform_indices=transform_indices,
            seed=seed,
            p=p,
            cls_idx=cls_idx,
            sample_fn=sample_fn,
        )
