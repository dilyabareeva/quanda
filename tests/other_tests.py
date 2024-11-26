import pytest
import torch

"""
Maybe we can find a different name/place for these tests.
"""


@pytest.mark.utils
def reproducibility_test():
    assert torch.__version__ == "2.0.0"
    gen = torch.Generator()
    gen.manual_seed(42)
    assert torch.all(
        torch.rand(5, generator=gen)
        == torch.Tensor([0.8823, 0.9150, 0.3829, 0.9593, 0.3904])
    )
