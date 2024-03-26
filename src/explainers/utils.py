import time

import torch


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
            print(f"{int(j / 1000)}/{int(len(self.dataset) / 1000)}k- 1000 images done in {mins} minutes {tdiff - 60 * mins}")
            t = time.time()
    return xpl
