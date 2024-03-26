import os
from typing import Union

import torch

from src.explainers.base import Explainer
from src.utils.data.feature_dataset import FeatureDataset


class FeatureKernelExplainer(Explainer):
    def __init__(
        self,
        model: torch.nn.Module,
        feature_extractor: Union[str, torch.nn.Module],
        classifier: Union[str, torch.nn.Module],
        dataset: torch.data.utils.Dataset,
        device: Union[str, torch.device],
        file: str,
        normalize: bool = True,
    ):
        super().__init__(model, dataset, device)
        # self.sanity_check = sanity_check
        if file is not None:
            if not os.path.isfile(file) and not os.path.isdir(file):
                file = None
        feature_ds = FeatureDataset(self.model, dataset, device, file)
        self.coefficients = None  # the coefficients for each training datapoint x class
        self.learned_weights = None
        self.normalize = normalize
        self.samples = feature_ds.samples.to(self.device)
        self.mean = self.samples.sum(0) / self.samples.shape[0]
        # self.mean = torch.zeros_like(self.mean)
        self.stdvar = torch.sqrt(torch.sum((self.samples - self.mean) ** 2, dim=0) / self.samples.shape[0])
        # self.stdvar=torch.ones_like(self.stdvar)
        self.normalized_samples = self.normalize_features(self.samples) if normalize else self.samples
        self.labels = torch.tensor(feature_ds.labels, dtype=torch.int, device=self.device)

    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.mean) / self.stdvar

    def explain(self, x: torch.Tensor, explanation_targets: torch.Tensor):
        assert self.coefficients is not None
        x = x.to(self.device)
        f = self.model.features(x)
        if self.normalize:
            f = self.normalize_features(f)
        crosscorr = torch.matmul(f, self.normalized_samples.T)
        crosscorr = crosscorr[:, :, None]
        xpl = self.coefficients * crosscorr
        indices = explanation_targets[:, None, None].expand(-1, self.samples.shape[0], 1)
        xpl = torch.gather(xpl, dim=-1, index=indices)
        return torch.squeeze(xpl)

    def save_coefs(self, dir: str):
        torch.save(self.coefficients, os.path.join(dir, f"{self.name}_coefs"))
