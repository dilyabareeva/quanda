from typing import Callable, List, Literal, Optional, Union

import torch
from PIL import Image

from quanda.utils.datasets.transformed import TransformedDataset

SampleFnLiterals = Literal["yellow_box"]


def yellow_box_sample_fn(img: Image.Image) -> Image.Image:
    """
    This function will add a yellow box to the image
    """
    square_size = (15, 15)  # Size of the square
    yellow_square = Image.new("RGB", square_size, (255, 255, 0))  # Create a yellow square
    img.paste(yellow_square, (10, 10))  # Paste it onto the image at the specified position
    return img


def get_sample_fn(name: SampleFnLiterals) -> Callable:
    if name == "yellow_box":
        return yellow_box_sample_fn
    else:
        raise ValueError(f"Unknown sample function {name}")


class SampleTransformationDataset(TransformedDataset):
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        n_classes: int,
        sample_fn: Union[SampleFnLiterals, Callable],
        dataset_transform: Optional[Callable] = None,
        transform_indices: Optional[List[int]] = None,
        cls_idx: Optional[int] = None,
        p: float = 1.0,
        seed: int = 42,
    ):
        if isinstance(sample_fn, str):
            sample_fn = get_sample_fn(sample_fn)
        elif not callable(sample_fn):
            raise ValueError("sample_fn should be a function or a valid literal")

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
