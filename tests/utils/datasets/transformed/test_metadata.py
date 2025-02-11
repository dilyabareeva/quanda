import pytest

from quanda.benchmarks.config_parser import BenchConfigParser
from quanda.utils.datasets.transformed import LabelFlippingDataset
from quanda.utils.datasets.transformed.metadata import LabelFlippingMetadata


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
def test_mislabeling_detection(
    test_id,
    config,
    expected_score,
    tmp_path,
    request,
):
    config = request.getfixturevalue(config)

    train_metadata = LabelFlippingMetadata(
        p=config["train_dataset"]["wrapper"]["metadata"]["p"],
        seed=config["train_dataset"]["wrapper"]["metadata"]["seed"],
    )

    dataset = LabelFlippingDataset(
        dataset=BenchConfigParser.process_dataset(
            dataset=config["train_dataset"]["dataset_str"],
            transform=None,
            dataset_split=config["train_dataset"]["dataset_split"],
        ),
        metadata=train_metadata,
    )
    train_metadata.mislabeling_labels = (
        train_metadata.generate_mislabeling_labels(dataset)
    )
    train_metadata_loaded = LabelFlippingMetadata.load(
        config["metadata_dir"],
        config["train_dataset"]["wrapper"]["metadata"]["metadata_filename"],
    )
    # assert metadata mislabeling_labels dicts are equal
    assert all(
        train_metadata.mislabeling_labels[key]
        == train_metadata_loaded.mislabeling_labels[key]
        for key in train_metadata.mislabeling_labels
    )
