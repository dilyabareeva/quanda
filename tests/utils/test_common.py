import pytest
import torch
import yaml

from quanda.utils.common import (
    DatasetSplit,
    _resolve_config,
    class_accuracy,
    get_targets,
    make_func,
)


@pytest.mark.utils
def test_resolve_config_dict_passes_through():
    cfg = {"id": "x", "bench": "ClassDetection"}
    assert _resolve_config(cfg) is cfg


@pytest.mark.utils
def test_resolve_config_registered_bench_id():
    """A registered ``bench_id`` resolves via ``config_map`` and parses."""
    cfg = _resolve_config("mnist_class_detection_unit")
    assert isinstance(cfg, dict)
    assert cfg.get("id")
    assert cfg.get("bench")


@pytest.mark.utils
def test_resolve_config_yaml_path(tmp_path):
    """An unregistered string is treated as a path to a YAML file."""
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"id": "from-disk", "k": 7}))
    cfg = _resolve_config(str(path))
    assert cfg == {"id": "from-disk", "k": 7}


@pytest.mark.utils
def test_resolve_config_rejects_non_mapping_yaml(tmp_path):
    """A YAML file that doesn't parse to a dict raises ``TypeError``."""
    path = tmp_path / "list.yaml"
    path.write_text(yaml.safe_dump([1, 2, 3]))
    with pytest.raises(TypeError, match="did not parse to a dict"):
        _resolve_config(str(path))


@pytest.mark.utils
@pytest.mark.parametrize("bad", [None, 42, 3.14, ["x"], object()])
def test_resolve_config_rejects_other_types(bad):
    with pytest.raises(TypeError, match="must be a dict"):
        _resolve_config(bad)


@pytest.mark.utils
def test_resolve_config_missing_path_raises(tmp_path):
    """An unregistered string that isn't a real file raises ``FileNotFoundError``."""
    missing = str(tmp_path / "does_not_exist.yaml")
    with pytest.raises(FileNotFoundError):
        _resolve_config(missing)


@pytest.mark.utils
@pytest.mark.parametrize(
    "test_id",
    [
        ("make_func",),
    ],
)
def test_trainer(
    test_id,
    request,
):
    def foo(x, y, z):
        return x * y + z

    bar = make_func(foo, {"x": 100, "z": 1})
    assert bar(y=1) == 101


@pytest.mark.utils
@pytest.mark.parametrize(
    "test_id, n_indices, test_size, val_size",
    [
        ("split", 100, 0.1, 0.1),
    ],
)
def test_train_test_val_split(
    test_id,
    n_indices,
    test_size,
    val_size,
    tmp_path,
    request,
):
    split = DatasetSplit.split(
        n_indices,
        24,
        {
            "test": test_size,
            "val": val_size,
            "train": 1 - test_size - val_size,
        },
    )

    split.save(str(tmp_path), "split.pt")

    loaded_split = DatasetSplit.load(str(tmp_path), "split.pt")

    train = loaded_split["train"]
    test = loaded_split["test"]
    val = loaded_split["val"]

    all = torch.cat([train, test, val]).sort().values

    assert torch.eq(all, torch.arange(n_indices)).all()


@pytest.mark.utils
@pytest.mark.parametrize(
    "test_id, model_type, labels, batch_size, single_class, expected",
    [
        (
            "tuple_all_correct",
            "tuple",
            [0, 0, 0, 0],
            4,
            None,
            1.0,
        ),
        (
            "tuple_all_wrong",
            "tuple",
            [1, 1, 1, 1],
            4,
            None,
            0.0,
        ),
        (
            "tuple_half_correct",
            "tuple",
            [0, 0, 1, 1],
            4,
            None,
            0.5,
        ),
        (
            "dict_with_logits_attr",
            "dict",
            [0, 0, 0, 0],
            4,
            None,
            1.0,
        ),
        (
            "dict_raw_tensor_output",
            "dict_raw",
            [0, 0, 0, 0],
            4,
            None,
            1.0,
        ),
        (
            "dict_half_correct",
            "dict",
            [0, 0, 1, 1],
            4,
            None,
            0.5,
        ),
        (
            "empty_loader",
            "tuple",
            [],
            1,
            None,
            0.0,
        ),
        (
            "single_class_override",
            "tuple",
            [1, 1, 1, 1],
            4,
            0,
            1.0,
        ),
        (
            "multi_batch_accumulation",
            "tuple",
            [0, 0, 1, 1, 0, 0],
            2,
            None,
            2.0 / 3.0,
        ),
    ],
)
def test_class_accuracy(
    test_id,
    model_type,
    labels,
    batch_size,
    single_class,
    expected,
    constant_model,
    constant_dict_model,
    tuple_dataloader,
    dict_dataloader,
):
    if model_type == "tuple":
        model = constant_model(predicted_class=0)
        loader = tuple_dataloader(labels, batch_size=batch_size)
    elif model_type == "dict":
        model = constant_dict_model(predicted_class=0, wrap_logits=True)
        loader = dict_dataloader(labels, batch_size=batch_size)
    else:
        model = constant_dict_model(predicted_class=0, wrap_logits=False)
        loader = dict_dataloader(labels, batch_size=batch_size)

    result = class_accuracy(model, loader, single_class=single_class)
    assert result == pytest.approx(expected)


@pytest.mark.utils
@pytest.mark.parametrize(
    "test_id, item, expected, error_match",
    [
        ("tuple_item", (torch.tensor([1.0]), 5), 5, None),
        (
            "dict_with_labels",
            {"input_ids": torch.tensor([1]), "labels": 3},
            3,
            None,
        ),
        (
            "dict_missing_labels",
            {"input_ids": torch.tensor([1])},
            None,
            "missing required 'labels' key",
        ),
        ("unsupported_type", [1, 2, 3], None, "Unsupported dataset item type"),
    ],
)
def test_get_targets(test_id, item, expected, error_match):
    if error_match is not None:
        with pytest.raises(ValueError, match=error_match):
            get_targets(item)
    else:
        assert get_targets(item) == expected
