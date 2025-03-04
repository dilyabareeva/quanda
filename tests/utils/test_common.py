import pytest
import torch

from quanda.utils.common import make_func, TrainValTest


@pytest.mark.utils
@pytest.mark.parametrize(
    "test_id",
    [
        ("make_func",),
    ],
)
def test_trainer(
    test_id,
    request,
):
    def foo(x, y, z):
        return x * y + z

    bar = make_func(foo, {"x": 100, "z": 1})
    assert bar(y=1) == 101


@pytest.mark.tested
@pytest.mark.parametrize(
    "test_id, n_indices, test_size, val_size",
    [
        ("split", 100, 0.1, 0.1),
    ],
)
def test_train_test_val_split(
    test_id,
    n_indices,
    test_size,
    val_size,
    tmp_path,
    request,
):
    split = TrainValTest.split(n_indices, 24, test_size, val_size)

    split.save(str(tmp_path), "split.pt")

    loaded_split = TrainValTest.load(str(tmp_path), "split.pt")

    train = loaded_split.train
    test = loaded_split.test
    val = loaded_split.val

    all = torch.cat([train, test, val]).sort().values

    assert torch.eq(all, torch.arange(n_indices)).all()
