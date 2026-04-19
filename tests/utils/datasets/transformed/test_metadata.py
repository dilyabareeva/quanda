import pytest
import torch
from torch.utils.data import TensorDataset

from quanda.benchmarks.config_parser import BenchConfigParser
from quanda.utils.datasets.transformed import LabelFlippingDataset
from quanda.utils.datasets.transformed.metadata import (
    ClassMapping,
    LabelFlippingMetadata,
    SampleTransformationMetadata,
)


@pytest.mark.utils
@pytest.mark.parametrize(
    "test_id, config, expected_score",
    [
        (
            "mnist",
            "load_mnist_mislabeling_config",
            0.4794672131538391,
        ),
    ],
)
def test_label_flipping_metadata(
    test_id,
    config,
    expected_score,
    tmp_path,
    request,
):
    config = request.getfixturevalue(config)

    metadata_dir = BenchConfigParser.get_metadata_dir(
        cfg=config, bench_save_dir=config.get("bench_save_dir", "./tmp")
    )

    train_metadata = LabelFlippingMetadata(
        p=config["train_dataset"]["wrapper"]["metadata"]["p"],
        seed=config["train_dataset"]["wrapper"]["metadata"]["seed"],
    )

    base_dataset = BenchConfigParser._parse_hf_dataset(
        dataset=config["train_dataset"]["dataset_str"],
        transform=None,
        dataset_split=config["train_dataset"]["dataset_split"],
    )
    base_dataset = BenchConfigParser._apply_indices(
        base_dataset,
        config["train_dataset"],
        metadata_dir,
        splits_cfg=config.get("splits", {}),
    )

    dataset = LabelFlippingDataset(
        dataset=base_dataset,
        metadata=train_metadata,
    )
    train_metadata.mislabeling_labels = (
        train_metadata.generate_mislabeling_labels(dataset)
    )

    train_metadata_loaded = LabelFlippingMetadata.load(
        metadata_dir,
        config["train_dataset"]["wrapper"]["metadata"]["metadata_filename"],
    )
    # assert metadata mislabeling_labels dicts are equal
    assert all(
        train_metadata.mislabeling_labels[key]
        == train_metadata_loaded.mislabeling_labels[key]
        for key in train_metadata.mislabeling_labels
    )


@pytest.mark.utils
def test_metadata_load_missing_raises(tmp_path):
    """load() raises FileNotFoundError when no metadata file exists."""
    with pytest.raises(FileNotFoundError, match="No metadata found"):
        LabelFlippingMetadata.load(str(tmp_path), "does_not_exist.yaml")


@pytest.mark.utils
@pytest.mark.parametrize(
    "test_id, kwargs, error_match",
    [
        ("invalid_prob", {"p": 1.5}, "Transformation probability"),
        (
            "invalid_indices",
            {"transform_indices": [0, 99]},
            "Invalid transform indices",
        ),
    ],
)
def test_metadata_validate_raises(test_id, kwargs, error_match):
    """SampleTransformationMetadata.validate rejects bad fields."""
    dataset = TensorDataset(torch.zeros(3, 1), torch.zeros(3))
    meta = SampleTransformationMetadata(**kwargs)
    with pytest.raises(ValueError, match=error_match):
        meta.validate(dataset)


@pytest.mark.utils
def test_class_mapping_resolve_integer_keys(tmp_path):
    """A spec with int keys is returned as a direct ClassMapping."""
    spec = {0: 0, 1: 1, 2: 0, 3: 1}
    mapping = ClassMapping.resolve(
        spec=spec, metadata_dir=str(tmp_path), load_meta_from_disk=False
    )
    assert mapping.class_to_group == spec
    assert mapping.n_classes == 4
    assert mapping.n_groups == 2
