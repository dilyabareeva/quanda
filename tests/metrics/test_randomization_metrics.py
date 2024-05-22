import pytest
import torch

from utils.explanations import TensorExplanations
from utils.functions.correlations import spearman_rank_corr
from metrics.randomization.model_randomization import ModelRandomizationMetric


@pytest.mark.randomize
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
    ModelRandomizationMetric._randomize_model(model2, gen)
    for param1, param2 in zip(model1.parameters(), model2.parameters()):
        assert torch.norm(param1.data - param2.data) > 1e3  # norm of the difference in parameters should be significant


@pytest.mark.randomize
def reproducibility_test():
    assert torch.__version__ == "2.0.0"
    gen = torch.Generator()
    gen.manual_seed(42)
    assert torch.all(torch.rand(5, generator=gen) == torch.Tensor([0.8823, 0.9150, 0.3829, 0.9593, 0.3904]))


@pytest.mark.randomize
def kendall_metric_test():
    def explain_fn(model):
        xpl_tensor = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        return TensorExplanations(xpl_tensor)

    xpl_tensor = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    metric = ModelRandomizationMetric(correlation_measure="kendall")
    assert torch.all(metric["rank_correlations"] == torch.tensor([1.0, -1.0]))


@pytest.mark.randomize
@pytest.mark.parametrize(
    "model",
    [
        ("load_mnist_model"),
    ],
)
def spearman_metric_test(model, request):
    def explain_fn(model):
        xpl_tensor = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        return TensorExplanations(xpl_tensor)

    def corr_measure(tensor1, tensor2):
        return spearman_rank_corr(tensor1, tensor2)

    model = request.getfixturevalue(model)
    xpl_tensor = torch.tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
    for corr_measure in ["spearman", "kendall", corr_measure]:
        metric = ModelRandomizationMetric(correlation_measure=corr_measure)
        metric = metric(model, "0", "", None, None, xpl_tensor, explain_fn, {})
        assert torch.all(metric["rank_correlations"] == torch.tensor([1.0, -1.0]))
