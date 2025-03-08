import os

import pytest
import torch

from quanda.explainers.wrappers import CaptumSimilarity
from quanda.utils.cache import BatchedCachedExplanations, ExplanationsCache
from quanda.utils.functions import cosine_similarity


@pytest.mark.utils
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, explanations, test_batches, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_explanations_similarity_1",
            "load_mnist_test_samples_batches",
            {"layers": "relu_4", "similarity_metric": cosine_similarity},
        ),
    ],
)
def test_batched_cached_explanations(
    test_id,
    model,
    checkpoint,
    dataset,
    explanations,
    test_batches,
    method_kwargs,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    dataset = request.getfixturevalue(dataset)
    test_batches = request.getfixturevalue(test_batches)

    explainer = CaptumSimilarity(
        model=model,
        checkpoints=checkpoint,
        model_id=test_id,
        cache_dir=str(tmp_path),
        train_dataset=dataset,
        device="cpu",
        **method_kwargs,
    )

    cache_path = os.path.join(str(tmp_path), "explanations_cache")
    os.mkdir(cache_path)

    # Produce explanations
    explanations = [
        explainer.explain(test_batch) for test_batch in test_batches
    ]

    # Save explanations to cache
    [
        ExplanationsCache.save(cache_path, xpl, i)
        for i, xpl in enumerate(explanations)
    ]

    # Load explanations from cache
    batched_explainer = BatchedCachedExplanations(
        cache_dir=cache_path, device="cpu"
    )
    loaded_explanations = [
        batched_explainer[i] for i in range(len(batched_explainer))
    ]

    comparison = [
        torch.allclose(loaded_explanations[i], explanations[i])
        for i in range(len(loaded_explanations))
    ]
    # Ensure cached explanations match the original expected explanations
    assert all([comparison]), "Cached explanations do not match expected"


@pytest.mark.utils
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, explanations, test_batches, method_kwargs",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_explanations_similarity_1",
            "load_mnist_test_samples_batches",
            {"layers": "relu_4", "similarity_metric": cosine_similarity},
        ),
    ],
)
def test_explanations_cache(
    test_id,
    model,
    checkpoint,
    dataset,
    explanations,
    test_batches,
    method_kwargs,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    dataset = request.getfixturevalue(dataset)
    test_batches = request.getfixturevalue(test_batches)

    explainer = CaptumSimilarity(
        model=model,
        checkpoints=checkpoint,
        model_id=test_id,
        cache_dir=str(tmp_path),
        train_dataset=dataset,
        device="cpu",
        **method_kwargs,
    )

    cache_path = os.path.join(str(tmp_path), "explanations_cache")
    os.mkdir(cache_path)

    cashew_path = os.path.join(str(tmp_path), "explanations_cashew")
    os.mkdir(cashew_path)

    # Produce explanations
    explanations = [
        explainer.explain(test_batch) for test_batch in test_batches
    ]

    # Save explanations to cache
    [
        ExplanationsCache.save(cache_path, xpl, i)
        for i, xpl in enumerate(explanations)
    ]

    assert (
        ExplanationsCache.exists(cache_path)
        & isinstance(
            ExplanationsCache.load(cache_path), BatchedCachedExplanations
        )
        & (not ExplanationsCache.exists(cashew_path))
    ), "Explanations cache not as expected."
