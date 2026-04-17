import copy
import os
from types import SimpleNamespace

import pytest
import torch

from quanda.utils.training.base_pl_module import BasicLightningModule
from quanda.utils.training.trainer import Trainer, _EpochSnapshotCallback


@pytest.mark.utils
@pytest.mark.parametrize(
    "test_id, init_model, dataloader, optimizer, lr, criterion, scheduler, scheduler_kwargs, \
    max_epochs, val_dataloaders, early_stopping, early_stopping_kwargs",
    [
        (
            "mnist",
            "load_init_mnist_model",
            "load_mnist_dataloader",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            "torch_constant_lr_scheduler_type",
            {"last_epoch": -1},
            3,
            "load_mnist_dataloader",
            False,
            {},
        ),
    ],
)
def test_trainer(
    test_id,
    init_model,
    dataloader,
    optimizer,
    lr,
    criterion,
    scheduler,
    scheduler_kwargs,
    max_epochs,
    val_dataloaders,
    early_stopping,
    early_stopping_kwargs,
    request,
):
    model = request.getfixturevalue(init_model)
    dataloader = request.getfixturevalue(dataloader)
    optimizer = request.getfixturevalue(optimizer)
    scheduler = request.getfixturevalue(scheduler)
    criterion = request.getfixturevalue(criterion)
    old_model = copy.deepcopy(model)

    trainer = Trainer(
        max_epochs=max_epochs,
        optimizer=optimizer,
        lr=lr,
        criterion=criterion,
        scheduler=scheduler,
        scheduler_kwargs=scheduler_kwargs,
    )

    model = trainer.fit(
        model=model,
        train_dataloaders=dataloader,
        val_dataloaders=dataloader,
        max_epochs=max_epochs,
    )

    for param1, param2 in zip(old_model.parameters(), model.parameters()):
        assert not torch.allclose(param1.data, param2.data), "Test failed."


@pytest.mark.utils
@pytest.mark.parametrize(
    "test_id, init_model, optimizer, lr, criterion, scheduler",
    [
        (
            "mnist",
            "load_init_mnist_model",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            lambda optimizer, **kwargs: "not_a_scheduler",
        ),
    ],
)
def test_pl_module_invalid_scheduler_raises(
    test_id,
    init_model,
    optimizer,
    lr,
    criterion,
    scheduler,
    request,
):
    model = request.getfixturevalue(init_model)
    optimizer = request.getfixturevalue(optimizer)
    criterion = request.getfixturevalue(criterion)

    pl_module = BasicLightningModule(
        model=model,
        optimizer=optimizer,
        lr=lr,
        criterion=criterion,
        scheduler=scheduler,
    )
    with pytest.raises(ValueError, match="scheduler must be an instance"):
        pl_module.configure_optimizers()


class _SnapshotModel:
    """Stand-in for a Hub-compatible model; records ``save_pretrained`` calls."""

    def __init__(self):
        self.save_calls = []

    def save_pretrained(self, path, safe_serialization=True):
        self.save_calls.append((path, safe_serialization))
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "marker.txt"), "w") as f:
            f.write("saved")


@pytest.mark.utils
def test_epoch_snapshot_callback_saves_at_configured_epochs(tmp_path):
    snapshot_epochs = [0, 2]
    snapshot_dirs = [
        str(tmp_path / "epoch_1"),
        str(tmp_path / "epoch_2"),
    ]
    callback = _EpochSnapshotCallback(snapshot_epochs, snapshot_dirs)

    model = _SnapshotModel()
    pl_module = SimpleNamespace(model=model)

    for epoch in range(4):
        callback.on_train_epoch_end(
            trainer=SimpleNamespace(current_epoch=epoch),
            pl_module=pl_module,
        )

    assert [call[0] for call in model.save_calls] == snapshot_dirs
    for d in snapshot_dirs:
        assert os.path.exists(os.path.join(d, "marker.txt"))
