import pytest
import torch

from quanda.explainers.wrappers.captum_influence import CaptumSimilarity
from quanda.tasks import ProponentsPerSample
from quanda.utils.functions.similarities import cosine_similarity


@pytest.mark.tasks
@pytest.mark.parametrize(
    "test_id, model, dataset, " "global_method, batch_size, explainer_cls, expl_kwargs, explanations ",
    [
        (
            "mnist1",
            "load_mnist_model",
            "load_mnist_dataset",
            "self-influence",
            8,
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            "load_mnist_explanations_1",
        ),
    ],
)
def test_proponents_per_sample(
    test_id,
    model,
    dataset,
    global_method,
    batch_size,
    explainer_cls,
    expl_kwargs,
    explanations,
    tmp_path,
    request,
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    explanations = request.getfixturevalue(explanations)
    expl_kwargs = {**expl_kwargs}

    proponents = ProponentsPerSample(
        model=model,
        train_dataset=dataset,
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        top_k=1,
    )
    proponents.update(explanations)

    proponents = proponents.compute()

    # assert shape is (len(dataset), 1)
    assert proponents.shape == (explanations.shape[0], 1)


@pytest.mark.tasks
@pytest.mark.parametrize(
    "test_id, model, dataset, " "global_method, batch_size, explainer_cls, expl_kwargs, explanations",
    [
        (
            "mnist1",
            "load_mnist_model",
            "load_mnist_dataset",
            "self-influence",
            8,
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            "load_mnist_explanations_1",
        ),
    ],
)
def test_proponents_per_sample_loading_saving_resetting(
    test_id,
    model,
    dataset,
    global_method,
    batch_size,
    explainer_cls,
    expl_kwargs,
    explanations,
    tmp_path,
    request,
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)
    expl_kwargs = {**expl_kwargs}

    proponents = ProponentsPerSample(
        model=model,
        train_dataset=dataset,
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        top_k=1,
    )
    proponents.update(request.getfixturevalue(explanations))

    proponents_1 = proponents.compute()
    state_dict = proponents.state_dict()
    proponents.reset()
    proponents.load_state_dict(state_dict)
    proponents_2 = proponents.compute()

    assert torch.allclose(proponents_1, proponents_2)