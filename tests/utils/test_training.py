import copy

import pytest
import torch

from quanda.utils.training.base_pl_module import BasicLightningModule
from quanda.utils.training.trainer import Trainer


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
    with pytest.raises(ValueError):
        pl_module.configure_optimizers()
