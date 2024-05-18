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
    gen.initial_seed(42)
    MPRTMetric._randomize_model(model2, gen)
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        assert (
            torch.norm(param1.item() - param2.item()) > 1e3
        )  # norm of the difference in parameters should be significant
