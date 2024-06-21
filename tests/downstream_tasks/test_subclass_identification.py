import pytest

from downstream_tasks.subclass_identification import SubclassIdentification
from explainers.wrappers.captum_influence import captum_similarity_explain
from utils.functions.similarities import cosine_similarity


@pytest.mark.downstream_tasks
@pytest.mark.parametrize(
    "test_id, model, optimizer, lr, criterion, max_epochs, dataset, n_classes, n_groups, seed, test_labels, "
    "batch_size, explain, explain_kwargs, expected_score",
    [
        (
            "mnist",
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
            1.0,
        ),
    ],
)
def test_identical_subclass_metrics(
    test_id,
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
    request,
):
    model = request.getfixturevalue(model)
    optimizer = request.getfixturevalue(optimizer)
    criterion = request.getfixturevalue(criterion)
    dataset = request.getfixturevalue(dataset)

    dst_eval = SubclassIdentification()
    dst_eval.init_trainer_from_train_arguments(
        model=model,
        optimizer=optimizer,
        lr=lr,
        criterion=criterion,
    )
    score = dst_eval.evaluate(
        train_dataset=dataset,
        val_dataset=None,
        n_classes=n_classes,
        n_groups=n_groups,
        class_to_group="random",
        explain_fn=explain,
        explain_kwargs=explain_kwargs,
        trainer_kwargs={"max_epochs": max_epochs},
        cache_dir="./cache",
        model_id="default_model_id",
        run_id="default_subclass_identification",
        seed=seed,
        batch_size=batch_size,
        device="cpu",
    )
    assert score == expected_score