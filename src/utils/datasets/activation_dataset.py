import glob
import os
from typing import Tuple, Union

import torch
from torch import Tensor
from torch.utils.data.dataset import Dataset


class ActivationDataset(Dataset):
    def __init__(self, layer_dir, device="cpu"):
        self.device = device
        self.av_filesearch = os.path.join(layer_dir, "*.pt")

        self.files = glob.glob(self.av_filesearch)

    def __getitem__(self, idx: int) -> Union[Tensor, Tuple[Tensor, ...]]:
        assert idx < len(self.files), "Layer index is out of bounds!"
        fl = self.files[idx]
        av = torch.load(fl, map_location=self.device)
        return av

    def __len__(self) -> int:
        return len(self.files)

    @property
    def samples_and_labels(self) -> Tuple[Tensor, Tensor]:
        samples = []
        labels = []

        for sample, label in self:
            samples.append(sample)
            labels.append(label)

        return torch.cat(samples).to(self.device), torch.cat(labels).to(self.device)
