import math

import pytest

from quanda.benchmarks.heuristics.mixed_datasets import MixedDatasets
from quanda.explainers.wrappers import CaptumSimilarity
from quanda.utils.datasets.image_datasets import SingleClassImageDataset
from quanda.utils.functions import cosine_similarity
from quanda.utils.training import Trainer


@pytest.mark.benchmarks
@pytest.mark.tested
@pytest.mark.parametrize(
    "test_id, init_method, model, optimizer, lr, criterion, max_epochs, dataset, adversarial_path,"
    "adversarial_indices, adversarial_label, adversarial_transforms, batch_size, explainer_cls, expl_kwargs,"
    "expected_score",
    [
        (
            "mnist",
            "generate",
            "load_mnist_model",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            3,
            "load_mnist_dataset",
            "load_fashion_mnist_path",
            "load_mnist_adversarial_indices",
            1,
            "load_fashion_mnist_to_mnist_transform",
            8,
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
                "model_id": "mnist",
            },
            0.8064,
        ),
        (
            "mnist",
            "assemble",
            "load_mnist_model",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            3,
            "load_mnist_dataset",
            "load_fashion_mnist_path",
            "load_mnist_adversarial_indices",
            1,
            "load_fashion_mnist_to_mnist_transform",
            8,
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
                "model_id": "mnist",
            },
            0.6500,
        ),
    ],
)
def test_mixed_datasets(
    test_id,
    init_method,
    model,
    optimizer,
    lr,
    criterion,
    max_epochs,
    dataset,
    adversarial_path,
    adversarial_indices,
    adversarial_label,
    adversarial_transforms,
    batch_size,
    explainer_cls,
    expl_kwargs,
    expected_score,
    tmp_path,
    request,
):
    model = request.getfixturevalue(model)
    optimizer = request.getfixturevalue(optimizer)
    criterion = request.getfixturevalue(criterion)
    dataset = request.getfixturevalue(dataset)
    adversarial_indices = request.getfixturevalue(adversarial_indices)
    adversarial_transforms = request.getfixturevalue(adversarial_transforms)
    adversarial_path = request.getfixturevalue(adversarial_path)

    if init_method == "generate":
        trainer = Trainer(
            max_epochs=max_epochs,
            optimizer=optimizer,
            lr=lr,
            criterion=criterion,
        )

        dst_eval = MixedDatasets.generate(
            model=model,
            trainer=trainer,
            clean_dataset=dataset,
            adversarial_label=adversarial_label,
            adversarial_dir=adversarial_path,
            adversarial_transform=adversarial_transforms,
            trainer_fit_kwargs={},
            batch_size=batch_size,
        )

    elif init_method == "assemble":
        dst_eval = MixedDatasets.assemble(
            model=model,
            train_dataset=dataset,
            adversarial_indices=adversarial_indices,
        )
    else:
        raise ValueError(f"Invalid init_method: {init_method}")

    eval_dataset = SingleClassImageDataset(root=adversarial_path, label=adversarial_label, transform=adversarial_transforms)

    score = dst_eval.evaluate(
        eval_dataset=eval_dataset,
        explainer_cls=explainer_cls,
        expl_kwargs={**expl_kwargs, "cache_dir": str(tmp_path)},
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)
