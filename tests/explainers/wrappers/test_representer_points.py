import os

import pytest
import torch

from quanda.explainers.wrappers import RepresenterPoints


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, checkpoint,dataset, test_data, test_labels, method_kwargs, use_postprocess",
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
            False,
        ),
        (
            "mnist_representer_postprocess",
            "load_mnist_model",
            "load_mnist_last_checkpoint",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            {
                "model_id": "pp",
                "batch_size": 8,
                "features_layer": "relu_4",
                "classifier_layer": "fc_3",
                "show_progress": False,
                "epoch": 2,
            },
            True,
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
    use_postprocess,
    request,
    tmp_path,
):
    model = request.getfixturevalue(model)
    checkpoint = request.getfixturevalue(checkpoint)
    dataset = request.getfixturevalue(dataset)
    test_data = request.getfixturevalue(test_data)
    test_labels = request.getfixturevalue(test_labels)

    calls = {"n": 0}

    def postprocess(feat):
        calls["n"] += 1
        return feat * 2.0

    kwargs = dict(method_kwargs)
    if use_postprocess:
        kwargs["features_postprocess"] = postprocess

    explainer = RepresenterPoints(
        model=model,
        checkpoints=checkpoint,
        cache_dir=str(tmp_path),
        train_dataset=dataset,
        **kwargs,
    )

    init_calls = calls["n"]
    if use_postprocess:
        assert init_calls >= 1

    explanations = explainer.explain(test_data=test_data, targets=test_labels)

    assert explanations.shape[0] == len(test_labels), (
        "Explanations shape is not as expected"
    )
    if use_postprocess:
        assert calls["n"] > init_calls


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
def test_representer_points_normalize_features(
    load_mnist_model,
    load_mnist_last_checkpoint,
    load_mnist_dataset,
    tmp_path,
    monkeypatch,
):
    """normalize=True exercises the _normalize_features path at init."""
    monkeypatch.setattr(
        RepresenterPoints,
        "train",
        lambda self: setattr(
            self, "coefficients", torch.zeros(len(load_mnist_dataset), 10)
        ),
    )

    explainer = RepresenterPoints(
        model=load_mnist_model,
        checkpoints=load_mnist_last_checkpoint,
        cache_dir=str(tmp_path),
        train_dataset=load_mnist_dataset,
        model_id="norm",
        batch_size=8,
        features_layer="relu_4",
        classifier_layer="fc_3",
        show_progress=False,
        normalize=True,
        load_from_disk=False,
    )
    assert explainer.normalize is True
    probe = torch.ones_like(explainer.mean) * 3.0 + explainer.mean
    normalized = explainer._normalize_features(probe)
    expected = torch.ones_like(probe) * 3.0 / explainer.std_dev
    finite = torch.isfinite(expected) & torch.isfinite(normalized)
    assert torch.allclose(normalized[finite], expected[finite], atol=1e-5)


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, load_from_disk, seed_bogus_cache",
    [
        ("cache_shape_mismatch", True, True),
        ("load_from_disk_false", False, False),
    ],
)
def test_representer_points_train_triggered(
    test_id,
    load_from_disk,
    seed_bogus_cache,
    load_mnist_model,
    load_mnist_last_checkpoint,
    load_mnist_dataset,
    tmp_path,
    monkeypatch,
):
    """train() is invoked when the cache is unusable or bypassed."""
    model_id = "0"
    if seed_bogus_cache:
        weights_path = os.path.join(tmp_path, f"{model_id}_repr_weights.pt")
        torch.save(torch.zeros(3, 10), weights_path)

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
        load_from_disk=load_from_disk,
    )

    assert call_count["n"] == 1
    assert explainer.coefficients.shape[0] == len(load_mnist_dataset)
