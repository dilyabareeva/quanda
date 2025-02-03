import torch
from typing import Optional, List
from lightning import LightningModule

import os
from quanda.benchmarks.resources.modules import bench_load_state_dict


def run_model_sanity_checks(
    ckpt_path: str,
    ckpt_names: List[str],
    pl_module: LightningModule,
    dataset_type: str,
    train_set: torch.utils.data.Dataset,
    val_set: torch.utils.data.Dataset,
    ds_dict: dict,
    device: str,
):
    sanity_checks = {
        "vanilla": [],
        "subclass": [],
        "mislabeled": [
            ("mislabeled_memorization", mislabeled_memorization_score)
        ],
        "mixed": [
            ("adversarial_memorization", adversarial_memorization_score),
            ("adversarial_classification", adversarial_classification_score),
        ],
        "shortcut": [
            ("shortcut_memorization", shortcut_memorization_score),
            ("shortcut_classification", shortcut_classification_score),
        ],
    }
    func_params = {
        "mislabeled_memorization": {
            "train_set": train_set,
            "mislabeling_indices": ds_dict.get("mislabeling_indices", None),
        },
        "adversarial_memorization": {
            "train_set": train_set,
            "adversarial_cls": ds_dict.get("adversarial_cls", None),
        },
        "adversarial_classification": {
            "score_set": ds_dict.get("adversarial_test_dataset", None),
            "adversarial_cls": ds_dict.get("adversarial_cls", None),
        },
        "shortcut_memorization": {
            "train_set": train_set,
            "shortcut_indices": ds_dict.get("shortcut_indices", None),
            "shortcut_cls": ds_dict.get("shortcut_cls", None),
        },
        "shortcut_classification": {
            "score_set": ds_dict.get("shortcut_val_dataset", None),
            "shortcut_cls": ds_dict.get("shortcut_cls", None),
        },
    }

    ckpt_names = sorted(
        ckpt_names, key=lambda x: int(x.split("=")[1].split(".")[0])
    )

    funcs = sanity_checks[dataset_type]
    ret_dict = {name: {"epochs": [], "values": []} for name, _ in funcs}

    for ckpt in ckpt_names:
        state_dict = torch.load(
            os.path.join(ckpt_path, ckpt), map_location=device
        )
        pl_module = bench_load_state_dict(pl_module, state_dict)
        epoch = int(ckpt.split("=")[1].split(".")[0])
        for name, f in funcs:
            ret_dict[name]["epochs"].append(epoch)
            kwargs = func_params[name]
            kwargs["model"] = pl_module.model
            kwargs["device"] = device
            ret_dict[name]["values"].append(f(**kwargs))
    return ret_dict


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
