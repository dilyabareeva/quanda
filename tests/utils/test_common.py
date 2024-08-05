import pytest

from quanda.utils.common import make_func


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
