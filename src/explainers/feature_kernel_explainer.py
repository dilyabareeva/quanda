import os
from typing import Union

import torch

from src.explainers.base import Explainer
from src.utils.cache import ActivationsCache as AC


class FeatureKernelExplainer(Explainer):
    def __init__(
        self,
        model: torch.nn.Module,
        layer: str,
        dataset: torch.data.utils.Dataset,
        device: Union[str, torch.device],
        file_path: str,
        normalize: bool = True,
    ):
        """

        :param model:
        :param dataset:
        :param device:
        :param file_path:
        :param normalize:
        """
        super().__init__(model, dataset, device)

        layer = "features"  # TODO: should be configurable
        self.coefficients = None  # the coefficients for each training datapoint x class
        self.learned_weights = None
        self.normalize = normalize

        self.samples, self.labels = self.generate_features(model, dataset, layer, file_path)
        self.mean = self.samples.sum(0) / self.samples.shape[0]
        self.stdvar = torch.sqrt(torch.sum((self.samples - self.mean) ** 2, dim=0) / self.samples.shape[0])
        self.normalized_samples = self.normalize_features(self.samples) if self.normalize else self.samples

    @staticmethod
    def generate_features(model, dataset, layer, file_path):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
        av_dataset = AC.generate_dataset_activations(
            path=file_path,
            model=model,
            layers=[layer],
            dataloader=dataloader,
            load_from_disk=True,
            return_activations=True,
        )[0]
        return av_dataset.samples_and_labels

    def normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        return (features - self.mean) / self.stdvar

    def explain(self, x: torch.Tensor, explanation_targets: torch.Tensor):
        assert self.coefficients is not None  # TODO: shouldn't we calculate coefficients in here?
        x = x.to(self.device)
        f = self.model.features(x)  # TODO: make it more flexible wrt. layer name
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
