import torch
from typing import Optional
from lightning import LightningModule


def accuracy(
    model: LightningModule,
    device: str,
    dataset: torch.utils.data.Dataset,
    targets: Optional[torch.Tensor] = None,
):
    BATCH_SIZE = 32
    model.to(device)
    if targets is not None:
        assert len(targets) == len(dataset)
        targets = targets.to(device)
    ld = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
    success_arr = torch.empty((0,), device=device)
    for i, (x, y) in enumerate(ld):
        x = x.to(device)
        y = y.to(device)
        target = y
        if targets is not None:
            target = targets[i * 32 : i * 32 + x.shape[0]]
        pred = model(x).argmax(dim=-1)
        success_arr = torch.concat((success_arr, pred == target))
    return torch.mean(success_arr * 1.0)


def mislabeled_memorization_score(
    model, train_set, mislabeling_indices, device
):
    score_set = torch.utils.data.Subset(train_set, mislabeling_indices)
    return accuracy(model, device, score_set)


def adversarial_memorization_score(model, train_set, adversarial_cls, device):
    score_set = train_set.datasets[0]
    return adversarial_classification_score(
        model=model,
        score_set=score_set,
        adversarial_cls=adversarial_cls,
        device=device,
    )


def adversarial_classification_score(
    model, score_set, adversarial_cls, device
):
    assert all([y == adversarial_cls for _, y in score_set])
    return accuracy(model, device, score_set)


def shortcut_memorization_score(
    model, train_set, shortcut_indices, shortcut_cls, device
):
    score_set = torch.utils.data.Subset(train_set, indices=shortcut_indices)
    assert all([y == shortcut_cls for _, y in score_set])
    return accuracy(model, device, score_set)


def shortcut_classification_score(model, score_set, shortcut_cls, device):
    targets = torch.tensor(
        [shortcut_cls for _ in range(len(score_set))], dtype=int
    )
    return accuracy(model, device, score_set, targets)
