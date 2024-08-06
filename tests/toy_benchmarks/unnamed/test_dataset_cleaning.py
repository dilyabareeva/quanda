import pytest

from src.explainers.wrappers.captum_influence import CaptumSimilarity
from src.toy_benchmarks.unnamed.dataset_cleaning import DatasetCleaning
from src.utils.functions.similarities import cosine_similarity
from src.utils.training.base_pl_module import BasicLightningModule
from src.utils.training.trainer import Trainer


@pytest.mark.toy_benchmarks
@pytest.mark.parametrize(
    "test_id, init_method, model, optimizer, lr, criterion, max_epochs, dataset, n_classes, n_groups, seed, "
    "global_method, batch_size, explainer_cls, expl_kwargs, use_pred, load_path, expected_score",
    [
        (
            "mnist1",
            "generate",
            "load_mnist_model",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            3,
            "load_mnist_dataset",
            10,
            2,
            27,
            "self-influence",
            8,
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            False,
            None,
            0.0,
        ),
        (
            "mnist2",
            "assemble",
            "load_mnist_grouped_model",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            3,
            "load_mnist_dataset",
            10,
            2,
            27,
            "sum",
            8,
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            False,
            None,
            -0.875,
        ),
        (
            "mnist3",
            "load",
            "load_mnist_model",
            "torch_sgd_optimizer",
            0.01,
            "torch_cross_entropy_loss_object",
            3,
            "load_mnist_dataset",
            10,
            2,
            27,
            "sum_abs",
            8,
            CaptumSimilarity,
            {
                "layers": "fc_2",
                "similarity_metric": cosine_similarity,
            },
            False,
            "tests/assets/mnist_dataset_cleaning_state_dict",
            0.0,
        ),
    ],
)
def test_dataset_cleaning(
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
    global_method,
    batch_size,
    explainer_cls,
    expl_kwargs,
    use_pred,
    load_path,
    expected_score,
    tmp_path,
    request,
):
    model = request.getfixturevalue(model)
    optimizer = request.getfixturevalue(optimizer)
    criterion = request.getfixturevalue(criterion)
    dataset = request.getfixturevalue(dataset)

    if init_method == "generate":
        dst_eval = DatasetCleaning.generate(
            model=model,
            train_dataset=dataset,
            device="cpu",
        )
        dst_eval.save("tests/assets/mnist_dataset_cleaning_state_dict")

    elif init_method == "load":
        dst_eval = DatasetCleaning.load(path=load_path)

    elif init_method == "assemble":
        dst_eval = DatasetCleaning.assemble(
            model=model,
            train_dataset=dataset,
        )
    else:
        raise ValueError(f"Invalid init_method: {init_method}")

    pl_module = BasicLightningModule(
        model=model,
        optimizer=optimizer,
        lr=lr,
        criterion=criterion,
    )

    trainer = Trainer.from_lightning_module(model, pl_module)
    score = dst_eval.evaluate(
        expl_dataset=dataset,
        explainer_cls=explainer_cls,
        trainer=trainer,
        expl_kwargs=expl_kwargs,
        trainer_fit_kwargs={"max_epochs": max_epochs},
        cache_dir=str(tmp_path),
        model_id="default_model_id",
        use_predictions=use_pred,
        global_method=global_method,
        batch_size=batch_size,
        device="cpu",
    )

    assert score == expected_score
