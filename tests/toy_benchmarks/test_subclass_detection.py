import pytest

from src.explainers.wrappers.captum_influence import captum_similarity_explain
from src.toy_benchmarks.subclass_detection import SubclassDetection
from src.utils.functions.similarities import cosine_similarity
from src.utils.training.base_pl_module import BasicLightningModule
from src.utils.training.trainer import Trainer


@pytest.mark.toy_benchmarks
@pytest.mark.parametrize(
    "test_id, init_method, model, optimizer, lr, criterion, max_epochs, dataset, n_classes, n_groups, seed, test_labels, "
    "batch_size, explain, explain_kwargs, expected_score",
    [
        (
            "mnist",
            "from_arguments",
            "load_mnist_model",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            3,
            "load_mnist_dataset",
            10,
            2,
            27,
            "load_mnist_test_labels_1",
            8,
            captum_similarity_explain,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            0.25,
        ),
        (
            "mnist",
            "from_pl",
            "load_mnist_model",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            3,
            "load_mnist_dataset",
            10,
            2,
            27,
            "load_mnist_test_labels_1",
            8,
            captum_similarity_explain,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            0.25,
        ),
        (
            "mnist",
            "from_trainer",
            "load_mnist_model",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            3,
            "load_mnist_dataset",
            10,
            2,
            27,
            "load_mnist_test_labels_1",
            8,
            captum_similarity_explain,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            0.25,
        ),
    ],
)
def test_subclass_detection(
    test_id,
    init_method,
    model,
    optimizer,
    lr,
    criterion,
    max_epochs,
    dataset,
    n_classes,
    n_groups,
    seed,
    test_labels,
    batch_size,
    explain,
    explain_kwargs,
    expected_score,
    tmp_path,
    request,
):
    model = request.getfixturevalue(model)
    optimizer = request.getfixturevalue(optimizer)
    criterion = request.getfixturevalue(criterion)
    dataset = request.getfixturevalue(dataset)

    if init_method == "from_arguments":

        dst_eval = SubclassDetection.generate(
            model=model,
            train_dataset=dataset,
            optimizer=optimizer,
            criterion=criterion,
            lr=lr,
            val_dataset=None,
            n_classes=n_classes,
            n_groups=n_groups,
            class_to_group="random",
            trainer_fit_kwargs={"max_epochs": max_epochs},
            seed=seed,
            batch_size=batch_size,
            device="cpu",
        )
    else:

        pl_module = BasicLightningModule(
            model=model,
            optimizer=optimizer,
            lr=lr,
            criterion=criterion,
        )

        if init_method == "from_pl":

            dst_eval = SubclassDetection.generate_from_pl(
                model=model,
                pl_module=pl_module,
                train_dataset=dataset,
                val_dataset=None,
                n_classes=n_classes,
                n_groups=n_groups,
                class_to_group="random",
                trainer_fit_kwargs={"max_epochs": max_epochs},
                seed=seed,
                batch_size=batch_size,
                device="cpu",
            )

        elif init_method == "from_trainer":

            trainer = Trainer.from_lightning_module(model, pl_module)

            dst_eval = SubclassDetection.generate_from_trainer(
                model=model,
                trainer=trainer,
                train_dataset=dataset,
                n_classes=n_classes,
                n_groups=n_groups,
                class_to_group="random",
                trainer_fit_kwargs={"max_epochs": max_epochs},
                seed=seed,
                batch_size=batch_size,
                device="cpu",
            )

        else:
            raise ValueError(f"Invalid init_method: {init_method}")

    score = dst_eval.evaluate(
        expl_dataset=dataset,
        explain_fn=explain,
        explain_kwargs=explain_kwargs,
        cache_dir=str(tmp_path),
        model_id="default_model_id",
        batch_size=batch_size,
        device="cpu",
    )

    assert score == expected_score
