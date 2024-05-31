import copy

import pytest
import torch

from utils.training.training import train_model


@pytest.mark.utils
@pytest.mark.parametrize(
    "test_id, init_model, dataloader, optimizer, criterion, max_epochs, val_loader, early_stopping, early_stopping_kwargs",
    [
        (
            "mnist",
            "load_init_mnist_model",
            "load_mnist_dataloader",
            "torch_sgd_optimizer",
            "torch_cross_entropy_loss_object",
            3,
            "load_mnist_dataloader",
            False,
            {},
        ),
    ],
)
def test_train_model(
    test_id,
    init_model,
    dataloader,
    optimizer,
    criterion,
    max_epochs,
    val_loader,
    early_stopping,
    early_stopping_kwargs,
    request,
):
    model = request.getfixturevalue(init_model)
    dataloader = request.getfixturevalue(dataloader)
    optimizer = request.getfixturevalue(optimizer)
    criterion = request.getfixturevalue(criterion)
    old_model = copy.deepcopy(model)
    model = train_model(
        model, dataloader, optimizer, criterion, max_epochs, val_loader, early_stopping, early_stopping_kwargs
    )

    for param1, param2 in zip(old_model.parameters(), model.parameters()):
        assert not torch.allclose(param1.data, param2.data), "Test failed."
