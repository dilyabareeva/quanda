from abc import ABC, abstractmethod
from typing import Union
import torch
import os


class Explainer(ABC):
    def __init__(self, model: torch.nn.Module, dataset: torch.data.utils.Dataset, device: Union[str, torch.device]):
        self.model = model
        self.device = torch.device(device) if isinstance(device, str) else device
        self.images = dataset
        self.samples = []
        self.labels = []
        dev = torch.device(device)
        self.model.to(dev)

    @abstractmethod
    def explain(self, x: torch.Tensor, explanation_targets: torch.Tensor) -> torch.Tensor:
        pass

    def train(self) -> None:
        pass

    def save_coefs(self, dir: str) -> None:
        pass


class FeatureKernelExplainer(Explainer):
    def __init__(
            self, model: torch.nn.Module, dataset: torch.data.utils.Dataset, device: Union[str, torch.device],
            file: str, normalize: bool = True
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


class GradientProductExplainer(Explainer):
    name = "GradientProductExplainer"

    def get_param_grad(self, x: torch.Tensor, index: int = None):
        x = x.to(self.device)
        out = self.model(x[None, :, :])
        if index is None:
            index = range(self.model.classifier.out_features)
        else:
            index = [index]
        grads = torch.empty(len(index), self.number_of_params)

        for i, ind in enumerate(index):
            assert ind > -1 and int(ind) == ind
            self.model.zero_grad()
            if self.loss is not None:
                out_new = self.loss(out, torch.eye(out.shape[1], device=self.device)[None, ind])
                out_new.backward(retain_graph=True)
            else:
                out[0][ind].backward(retain_graph=True)
            cumul = torch.empty(0, device=self.device)
            for par in self.model.sim_parameters():
                grad = par.grad.flatten()
                cumul = torch.cat((cumul, grad), 0)
            grads[i] = cumul

        return torch.squeeze(grads)

    def __init__(self, model:torch.nn.Module, dataset:torch.utils.data.Dataset, device:Union[str, torch.device], loss=None):
        super().__init__(model, dataset, device)
        self.number_of_params = 0
         self.loss = loss
        for p in list(self.model.sim_parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            self.number_of_params += nn
        # USE get_param_grad instead of grad_ds = GradientDataset(self.model, dataset)
        self.dataset = dataset

    def explain(self, x, preds=None, targets=None):
        assert not ((targets is None) and (self.loss is not None))
        xpl = torch.zeros((x.shape[0], len(self.dataset)), dtype=torch.float)
        xpl = xpl.to(self.device)
        t = time.time()
        for j in range(len(self.dataset)):
            tr_sample, y = self.dataset[j]
            train_grad = self.get_param_grad(tr_sample, y)
            train_grad = train_grad / torch.norm(train_grad)
            train_grad.to(self.device)
            for i in range(x.shape[0]):
                if self.loss is None:
                    test_grad = self.get_param_grad(x[i], preds[i])
                else:
                    test_grad = self.get_param_grad(x[i], targets[i])
                test_grad.to(self.device)
                xpl[i, j] = torch.matmul(train_grad, test_grad)
            if j % 1000 == 0:
                tdiff = time.time() - t
                mins = int(tdiff / 60)
                print(
                    f'{int(j / 1000)}/{int(len(self.dataset) / 1000)}k- 1000 images done in {mins} minutes {tdiff - 60 * mins}'
                )
                t = time.time()
        return xpl
