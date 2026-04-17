import os

import pytest
import torch

from quanda.explainers.wrappers import RepresenterPoints


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, test_data, test_labels, method_kwargs",
    [
        (
            "mnist_representer",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            {
                "model_id": "0",
                "batch_size": 8,
                "features_layer": "relu_4",
                "classifier_layer": "fc_3",
                "show_progress": False,
                "epoch": 5,
            },
        ),
    ],
)
def test_representer_points_explain(
    test_id,
    model,
    checkpoint,
    dataset,
    test_data,
    test_labels,
    method_kwargs,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    dataset = request.getfixturevalue(dataset)
    test_data = request.getfixturevalue(test_data)
    test_labels = request.getfixturevalue(test_labels)

    explainer = RepresenterPoints(
        model=model,
        checkpoints=checkpoint,
        cache_dir=str(tmp_path),
        train_dataset=dataset,
        **method_kwargs,
    )

    explanations = explainer.explain(test_data=test_data, targets=test_labels)

    assert explanations.shape[0] == len(test_labels), (
        "Explanations shape is not as expected"
    )


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, train_labels, method_kwargs",
    [
        (
            "mnist_representer",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_labels",
            {
                "model_id": "0",
                "batch_size": 8,
                "features_layer": "relu_4",
                "classifier_layer": "fc_3",
                "show_progress": False,
                "epoch": 5,
            },
        ),
    ],
)
def test_representer_points_self_influence(
    test_id,
    model,
    checkpoint,
    dataset,
    train_labels,
    method_kwargs,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    dataset = request.getfixturevalue(dataset)
    train_labels = request.getfixturevalue(train_labels)

    explainer = RepresenterPoints(
        model=model,
        checkpoints=checkpoint,
        cache_dir=str(tmp_path),
        train_dataset=dataset,
        **method_kwargs,
    )

    self_influence = explainer.self_influence()

    assert self_influence[7] == explainer.coefficients[7][train_labels[7]], (
        "Self-influence is not as expected"
    )


@pytest.mark.explainers
def test_representer_points_cached_shape_mismatch_retrains(
    load_mnist_model,
    load_mnist_last_checkpoint,
    load_mnist_dataset,
    tmp_path,
    monkeypatch,
):
    """Pre-seed a cached weights file with mismatched shape to
    exercise the retraining branch without running full training."""
    model_id = "0"
    weights_path = os.path.join(tmp_path, f"{model_id}_repr_weights.pt")
    bogus_coefficients = torch.zeros(3, 10)
    torch.save(bogus_coefficients, weights_path)

    call_count = {"n": 0}

    def fake_train(self):
        call_count["n"] += 1
        self.coefficients = torch.zeros(len(load_mnist_dataset), 10)

    monkeypatch.setattr(RepresenterPoints, "train", fake_train)

    explainer = RepresenterPoints(
        model=load_mnist_model,
        checkpoints=load_mnist_last_checkpoint,
        cache_dir=str(tmp_path),
        train_dataset=load_mnist_dataset,
        model_id=model_id,
        batch_size=8,
        features_layer="relu_4",
        classifier_layer="fc_3",
        show_progress=False,
    )

    assert call_count["n"] == 1
    assert explainer.coefficients.shape[0] == len(load_mnist_dataset)
