import copy

import pytest
import torch

from src.utils.training.base_pl_module import BasicLightningModule
from src.utils.training.trainer import Trainer


@pytest.mark.utils
@pytest.mark.parametrize(
    "test_id, init_model, dataloader, optimizer, lr, criterion, max_epochs, \
    val_loader, early_stopping, early_stopping_kwargs, mode",
    [
        (
            "mnist",
            "load_init_mnist_model",
            "load_mnist_dataloader",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            3,
            "load_mnist_dataloader",
            False,
            {},
            "from_arguments",
        ),
        (
            "mnist",
            "load_init_mnist_model",
            "load_mnist_dataloader",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            3,
            "load_mnist_dataloader",
            False,
            {},
            "from_pl_module",
        ),
    ],
)
def test_from_arguments(
    test_id,
    init_model,
    dataloader,
    optimizer,
    lr,
    criterion,
    max_epochs,
    val_loader,
    early_stopping,
    early_stopping_kwargs,
    mode,
    request,
):
    model = request.getfixturevalue(init_model)
    dataloader = request.getfixturevalue(dataloader)
    optimizer = request.getfixturevalue(optimizer)
    criterion = request.getfixturevalue(criterion)
    old_model = copy.deepcopy(model)
    trainer = Trainer()
    if mode == "from_arguments":
        trainer = trainer.from_arguments(
            model=model,
            optimizer=optimizer,
            lr=lr,
            criterion=criterion,
        )
    else:
        pl_module = BasicLightningModule(
            model=model,
            optimizer=optimizer,
            lr=lr,
            criterion=criterion,
        )
        trainer = trainer.from_lightning_module(model=model, pl_module=pl_module)
    model = trainer.fit(
        dataloader,
        dataloader,
        trainer_fit_kwargs={"max_epochs": max_epochs},
    )

    for param1, param2 in zip(old_model.parameters(), model.parameters()):
        assert not torch.allclose(param1.data, param2.data), "Test failed."
