import math
from functools import reduce

import pytest
import torch
from torcheval.metrics.functional import binary_auprc

from quanda.benchmarks.heuristics.mixed_datasets import MixedDatasets
from quanda.explainers.wrappers import CaptumSimilarity
from quanda.utils.datasets.image_datasets import SingleClassImageDataset
from quanda.utils.functions import cosine_similarity
from quanda.utils.training import Trainer


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, init_method, model, checkpoint, optimizer, lr, criterion, max_epochs, dataset, adversarial_path,"
    "adversarial_label, adversarial_transforms, adv_train_indices, adv_eval_indices,batch_size, explainer_cls, expl_kwargs,"
    "expected_score",
    [
        (
            "mnist_generate",
            "generate",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            3,
            "load_mnist_dataset",
            "load_fashion_mnist_path",
            3,
            "load_fashion_mnist_to_mnist_transform",
            None,
            None,
            8,
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
                "model_id": "mnist",
            },
            0.7923794984817505,
        ),
        (
            "mnist_assemble",
            "assemble",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            3,
            "load_mnist_dataset",
            "load_fashion_mnist_path",
            4,
            "load_fashion_mnist_to_mnist_transform",
            None,
            None,
            8,
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
                "model_id": "mnist",
            },
            0.8333333730697632,
        ),
    ],
)
def test_mixed_datasets(
    test_id,
    init_method,
    model,
    checkpoint,
    optimizer,
    lr,
    criterion,
    max_epochs,
    dataset,
    adversarial_path,
    adversarial_label,
    adversarial_transforms,
    adv_train_indices,
    adv_eval_indices,
    batch_size,
    explainer_cls,
    expl_kwargs,
    expected_score,
    tmp_path,
    request,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    optimizer = request.getfixturevalue(optimizer)
    criterion = request.getfixturevalue(criterion)
    dataset = request.getfixturevalue(dataset)
    adversarial_transforms = request.getfixturevalue(adversarial_transforms)
    adversarial_path = request.getfixturevalue(adversarial_path)

    eval_dataset = SingleClassImageDataset(
        root=adversarial_path,
        label=adversarial_label,
        transform=adversarial_transforms,
        indices=adv_eval_indices,
    )

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
            base_dataset=dataset,
            eval_dataset=eval_dataset,
            adversarial_label=adversarial_label,
            adversarial_dir=adversarial_path,
            adv_train_indices=adv_train_indices,
            adversarial_transform=adversarial_transforms,
            trainer_fit_kwargs={},
            cache_dir=str(tmp_path),
            batch_size=batch_size,
        )

    elif init_method == "assemble":
        dst_eval = MixedDatasets.assemble(
            model=model,
            checkpoints=checkpoint,
            base_dataset=dataset,
            eval_dataset=eval_dataset,
            adversarial_label=adversarial_label,
            adversarial_dir=adversarial_path,
            adversarial_transform=adversarial_transforms,
            adv_train_indices=adv_train_indices,
        )
    else:
        raise ValueError(f"Invalid init_method: {init_method}")

    score = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs={**expl_kwargs, "cache_dir": str(tmp_path)},
    )["score"]

    assert math.isclose(score, expected_score, abs_tol=0.00001)


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, benchmark, batch_size",
    [
        (
            "mnist",
            "mnist_mixed_datasets_benchmark",
            8,
        ),
    ],
)
def test_mixed_dataset_download_sanity_checks(
    test_id, benchmark, batch_size, request
):
    dst_eval = request.getfixturevalue(benchmark)
    nonadv_indices = torch.where(
        1 - torch.tensor(dst_eval.adversarial_indices)
    )[0]
    assertions = []
    assert len(nonadv_indices) == len(dst_eval.base_dataset)
    len_adv_ds = len(
        torch.where(torch.tensor(dst_eval.adversarial_indices))[0]
    )
    assertions.append(
        all([idx.item() >= len_adv_ds for idx in nonadv_indices[:500]])
    )

    assertions.append(
        all(
            [
                torch.allclose(
                    dst_eval.base_dataset[i.item() - len_adv_ds][0],
                    dst_eval.mixed_dataset[i.item()][0],
                )
                for i in nonadv_indices[:100]
            ]
        )
    )
    assert all(assertions)


@pytest.mark.benchmarks
@pytest.mark.parametrize(
    "test_id, benchmark, batch_size, explainer_cls, expl_kwargs, filter_by_prediction, expected_score",
    [
        (
            "mnist",
            "mnist_mixed_datasets_benchmark",
            8,
            CaptumSimilarity,
            {
                "layers": "model.fc_2",
                "similarity_metric": cosine_similarity,
                "load_from_disk": True,
            },
            True,
            "compute",
        ),
        (
            "mnist",
            "mnist_mixed_datasets_benchmark",
            8,
            CaptumSimilarity,
            {
                "layers": "model.fc_2",
                "similarity_metric": cosine_similarity,
                "load_from_disk": True,
            },
            False,
            "compute",
        ),
    ],
)
def test_mixed_dataset_download(
    test_id,
    benchmark,
    batch_size,
    explainer_cls,
    expl_kwargs,
    filter_by_prediction,
    expected_score,
    tmp_path,
    request,
):
    dst_eval = request.getfixturevalue(benchmark)

    expl_kwargs = {"model_id": "0", "cache_dir": str(tmp_path), **expl_kwargs}
    dst_eval.mixed_dataset = torch.utils.data.Subset(
        dst_eval.mixed_dataset, list(range(16))
    )
    dst_eval.eval_dataset = torch.utils.data.Subset(
        dst_eval.eval_dataset, list(range(8))
    )

    dst_eval.adversarial_indices = dst_eval.adversarial_indices[:16]
    dst_eval.filter_by_prediction = filter_by_prediction
    score = dst_eval.evaluate(
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        batch_size=batch_size,
    )["score"]

    if expected_score == "compute":
        activation = []

        def hook(model, input, output):
            activation.append(output.detach())

        exp_layer = reduce(
            getattr, expl_kwargs["layers"].split("."), dst_eval.model
        )
        exp_layer.register_forward_hook(hook)
        train_ld = torch.utils.data.DataLoader(
            dst_eval.mixed_dataset, batch_size=16, shuffle=False
        )
        test_ld = torch.utils.data.DataLoader(
            dst_eval.eval_dataset, batch_size=8, shuffle=False
        )
        for x, y in iter(train_ld):
            x = x.to(dst_eval.device)
            dst_eval.model(x)
        act_train = activation[0]
        activation = []
        for x, y in iter(test_ld):
            x = x.to(dst_eval.device)
            y_preds = dst_eval.model(x).argmax(dim=-1)
            select_idx = torch.tensor([True] * 8)
            if filter_by_prediction:
                select_idx *= y_preds == dst_eval.adversarial_label
            dst_eval.model(x)
        act_test = activation[0]
        act_test = act_test[select_idx]
        act_test = torch.nn.functional.normalize(act_test, dim=-1)
        act_train = torch.nn.functional.normalize(act_train, dim=-1)
        IP = torch.matmul(act_test, act_train.T)
        expected_score = (
            torch.tensor(
                [
                    binary_auprc(
                        xpl, torch.tensor(dst_eval.adversarial_indices)
                    )
                    for xpl in IP
                ]
            )
            .mean()
            .item()
        )

    assert math.isclose(score, expected_score, abs_tol=0.00001)
