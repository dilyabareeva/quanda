import time
from typing import Union

import torch

from src.explainers.base import Explainer


class GradientProductExplainer(Explainer):
    name = "GradientProductExplainer"

    def __init__(
        self, model: torch.nn.Module, dataset: torch.utils.data.Dataset, device: Union[str, torch.device], loss=None
    ):
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
                    f"{int(j / 1000)}/{int(len(self.dataset) / 1000)}k- 1000 images done in {mins} minutes {tdiff - 60 * mins}"
                )
                t = time.time()
        return xpl
