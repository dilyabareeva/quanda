import pytest

from src.explainers.aggregators import SumAggregator
from src.explainers.wrappers.captum_influence import CaptumSimilarity
from src.toy_benchmarks.mislabeling_detection import MislabelingDetection
from src.utils.functions.similarities import cosine_similarity
from src.utils.training.base_pl_module import BasicLightningModule
from src.utils.training.trainer import Trainer


@pytest.mark.toy_benchmarks
@pytest.mark.parametrize(
    "test_id, init_method, model, optimizer, lr, criterion, max_epochs, dataset, n_classes, p, seed, test_labels, "
    "global_method, batch_size, explainer_cls, expl_kwargs, expected_score",
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
            1.0,
            27,
            "load_mnist_test_labels_1",
            "self-influence",
            8,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity, "cache_dir": "cache", "model_id": "test"},
            0.5625,
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
            1.0,
            27,
            "load_mnist_test_labels_1",
            "sum_abs",
            8,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity, "cache_dir": "cache", "model_id": "test"},
            0.5625,
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
            1.0,
            27,
            "load_mnist_test_labels_1",
            SumAggregator,
            8,
            CaptumSimilarity,
            {"layers": "fc_2", "similarity_metric": cosine_similarity, "cache_dir": "cache", "model_id": "test"},
            0.5625,
        ),
    ],
)
def test_mislabeling_detection(
    test_id,
    init_method,
    model,
    optimizer,
    lr,
    criterion,
    max_epochs,
    dataset,
    n_classes,
    p,
    seed,
    test_labels,
    batch_size,
    global_method,
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

    if init_method == "from_arguments":

        dst_eval = MislabelingDetection.generate(
            model=model,
            train_dataset=dataset,
            optimizer=optimizer,
            criterion=criterion,
            lr=lr,
            val_dataset=None,
            n_classes=n_classes,
            p=p,
            global_method=global_method,
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

            dst_eval = MislabelingDetection.generate_from_pl(
                model=model,
                pl_module=pl_module,
                train_dataset=dataset,
                val_dataset=None,
                n_classes=n_classes,
                p=p,
                global_method=global_method,
                class_to_group="random",
                trainer_fit_kwargs={"max_epochs": max_epochs},
                seed=seed,
                batch_size=batch_size,
                device="cpu",
            )

        elif init_method == "from_trainer":

            trainer = Trainer.from_lightning_module(model, pl_module)

            dst_eval = MislabelingDetection.generate_from_trainer(
                model=model,
                trainer=trainer,
                train_dataset=dataset,
                n_classes=n_classes,
                p=p,
                global_method=global_method,
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
        explainer_cls=explainer_cls,
        expl_kwargs=expl_kwargs,
        cache_dir=str(tmp_path),
        model_id="default_model_id",
        batch_size=batch_size,
        device="cpu",
    )["score"]

    assert score == expected_score
