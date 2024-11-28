import math

import pytest
import torch

from quanda.benchmarks.ground_truth.linear_datamodeling import (
    LinearDatamodeling,
)
from quanda.explainers.wrappers.captum_influence import CaptumSimilarity
from quanda.utils.functions.correlations import spearman_rank_corr
from quanda.utils.functions.similarities import cosine_similarity
from quanda.utils.training.trainer import Trainer


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, init_method, model, checkpoint, optimizer, lr, criterion, dataset, n_classes, seed, "
    "batch_size, explainer_cls, expl_kwargs, use_pred",
    [
        (
            "mnist0",
            "generate",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            "load_mnist_dataset",
            10,
            27,
            8,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            False,
        ),
        (
            "mnist1",
            "assemble",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            "load_mnist_dataset",
            10,
            27,
            8,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            False,
        ),
        (
            "mnist2",
            "assemble",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            "load_mnist_dataset",
            10,
            27,
            8,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity},
            True,
        ),
    ],
)
def test_linear_datamodeling(
    test_id,
    init_method,
    model,
    checkpoint,
    optimizer,
    lr,
    criterion,
    dataset,
    n_classes,
    seed,
    batch_size,
    explainer_cls,
    expl_kwargs,
    use_pred,
    tmp_path,
    request,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    optimizer = request.getfixturevalue(optimizer)
    criterion = request.getfixturevalue(criterion)
    dataset = request.getfixturevalue(dataset)

    expl_kwargs = {
        **expl_kwargs,
        "model_id": test_id,
        "cache_dir": str(tmp_path),
    }
    trainer = Trainer(
        max_epochs=0,
        optimizer=optimizer,
        lr=lr,
        criterion=criterion,
    )
    if init_method == "generate":
        benchmark = LinearDatamodeling.generate(
            model=model,
            checkpoints=checkpoint,
            trainer=trainer,
            train_dataset=dataset,
            eval_dataset=dataset,
            n_classes=n_classes,
            seed=seed,
            batch_size=batch_size,
            use_predictions=use_pred,
            cache_dir=str(tmp_path),
            model_id=test_id,
            m=5,
            alpha=0.5,
            correlation_fn="spearman",
            device="cpu",
        )

    elif init_method == "assemble":
        benchmark = LinearDatamodeling.assemble(
            model=model,
            checkpoints=checkpoint,
            trainer=trainer,
            train_dataset=dataset,
            eval_dataset=dataset,
            n_classes=n_classes,
            seed=seed,
            batch_size=batch_size,
            use_predictions=use_pred,
            cache_dir=str(tmp_path),
            model_id=test_id,
            m=5,
            alpha=0.5,
            correlation_fn="spearman",
            device="cpu",
        )
    else:
        raise ValueError(f"Invalid init_method: {init_method}")

    score = benchmark.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=batch_size,
    )["score"]

    # compute expected score by hand:
    gen = torch.Generator()
    gen.manual_seed(seed)
    subsets = []
    for i in range(5):
        subset = torch.randperm(len(dataset), generator=gen)[
            : int(len(dataset) * 0.5)
        ]
        subsets.append(subset)

    expl = explainer_cls(
        model=model,
        checkpoints=checkpoint,
        train_dataset=dataset,
        **expl_kwargs,
    )
    ldr = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    explanations = torch.empty((0, len(dataset)))
    outputs = torch.empty((0, len(subsets)))
    for x, y in ldr:
        output = model(x)
        if use_pred:
            target = output.argmax(dim=1)
        else:
            target = y
        explanations = torch.cat(
            (explanations, expl.explain(x, target)), dim=0
        )
        output = torch.tensor(
            [output[i, target[i]] for i in range(batch_size)]
        )
        output = torch.vstack([output for i in range(len(subsets))]).T
        outputs = torch.cat((outputs, output), dim=0)
    counterfactual = torch.stack(
        [explanations[:, subset.tolist()].sum(dim=1) for subset in subsets]
    ).T
    expected_score = spearman_rank_corr(outputs, counterfactual)
    expected_score = expected_score.mean().item()
    assert math.isclose(score, expected_score, abs_tol=0.00001)
