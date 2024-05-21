import pytest
import torch

from metrics.randomization.mprt import MPRTMetric


@pytest.mark.parametrize(
    "model",
    [
        ("load_mnist_model"),
    ],
)
def model_randomization_test(model, request):
    model1 = request.getfixturevalue(model)
    model2 = request.getfixturevalue(model)
    gen = torch.Generator()
    gen.manual_seed(42)
    MPRTMetric._randomize_model(model2, gen)
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        assert torch.norm(param1.data - param2.data) > 1e3  # norm of the difference in parameters should be significant


def model_randomization_test():
    assert torch.__version__=="2.0.0"
    gen = torch.Generator()
    gen.manual_seed(42)
    assert torch.all(torch.rand(5,generator=gen)==torch.Tensor([0.8823, 0.9150, 0.3829, 0.9593, 0.3904]))
