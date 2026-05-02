"""MNIST-related fixtures: data, LeNet models, checkpoints, configs."""

import json
import pickle

import numpy as np
import pytest
import torch
import torchvision
import yaml
from torch.utils.data import TensorDataset

from quanda.benchmarks.resources import config_map
from quanda.utils.datasets.transformed.label_flipping import (
    LabelFlippingDataset,
)
from quanda.utils.datasets.transformed.label_grouping import (
    LabelGroupingDataset,
)
from quanda.utils.datasets.transformed.metadata import ClassMapping
from quanda.utils.training.base_pl_module import BasicLightningModule
from tests.models import LeNet

MNIST_IMAGE_SIZE = 28
BATCH_SIZE = 124
MINI_BATCH_SIZE = 8


class TestTensorDataset(TensorDataset):
    """TensorDataset variant that returns (tensor, scalar-int) tuples."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = self.tensors[0]
        self._targets = self.tensors[1]

    def __getitem__(self, index):
        return self._data[index], self._targets[index].item()


@pytest.fixture
def mnist_range_explanations():
    return torch.tensor(
        [
            [i * 1.0 for i in range(8)],
            [i * 1.0 for i in range(8)],
            [i * 1.0 for i in range(8)],
        ],
        dtype=torch.float,
    )


@pytest.fixture
def range_ranking():
    return torch.tensor([i for i in range(8)])


@pytest.fixture
def mnist_seed_27_mislabeling_labels():
    with open(
        "tests/assets/dataset/mnist_seed_27_poisoned_labels.json",
        "r",
    ) as f:
        return json.load(f)


@pytest.fixture
def load_mnist_checkpoints():
    """Get paths to five checkpoints from a single training run."""
    return [
        "tests/assets/mnist_checkpoints/checkpoint-00",
        "tests/assets/mnist_checkpoints/checkpoint-01",
        "tests/assets/mnist_checkpoints/checkpoint-02",
        "tests/assets/mnist_checkpoints/checkpoint-03",
        "tests/assets/mnist_checkpoints/checkpoint-04",
    ]


@pytest.fixture
def load_mnist_model():
    """Load a pre-trained LeNet classification model."""
    model = LeNet()
    model.load_state_dict(
        torch.load(
            "tests/assets/mnist", map_location="cpu", pickle_module=pickle
        )
    )
    return model


@pytest.fixture
def load_mnist_model_with_custom_param():
    """Pre-trained LeNet with an extra custom parameter."""
    model = LeNet()
    model.load_state_dict(
        torch.load(
            "tests/assets/mnist", map_location="cpu", pickle_module=pickle
        )
    )
    model.custom_param = torch.nn.Parameter(torch.randn(4))
    return model


@pytest.fixture
def load_mnist_last_checkpoint():
    return "tests/assets/mnist"


@pytest.fixture
def load_mnist_pl_module():
    model = LeNet()
    model.load_state_dict(
        torch.load(
            "tests/assets/mnist", map_location="cpu", pickle_module=pickle
        )
    )
    return BasicLightningModule(
        model=model,
        optimizer=torch.optim.SGD,
        lr=0.01,
        criterion=torch.nn.CrossEntropyLoss(),
    )


@pytest.fixture
def load_mnist_grouped_model():
    model = LeNet(num_outputs=2)
    model.load_state_dict(
        torch.load(
            "tests/assets/mnist_grouped_model",
            map_location="cpu",
            pickle_module=pickle,
        )
    )
    return model


@pytest.fixture
def load_init_mnist_model():
    return LeNet()


def _load_mnist_xy() -> tuple:
    x_batch = (
        np.loadtxt("tests/assets/mnist_x")
        .astype(float)
        .reshape((BATCH_SIZE, 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE))
    )[:MINI_BATCH_SIZE]
    y_batch = np.loadtxt("tests/assets/mnist_y").astype(int)[:MINI_BATCH_SIZE]
    return x_batch, y_batch


@pytest.fixture
def load_mnist_dataset():
    x_batch, y_batch = _load_mnist_xy()
    return TestTensorDataset(
        torch.tensor(x_batch).float(), torch.tensor(y_batch).long()
    )


@pytest.fixture
def load_mnist_labels():
    _, y_batch = _load_mnist_xy()
    return torch.tensor(y_batch).long()


@pytest.fixture
def load_mnist_adversarial_indices():
    _, y_batch = _load_mnist_xy()
    return [int(y == 1) for y in y_batch]


@pytest.fixture
def load_grouped_mnist_dataset():
    x_batch, y_batch = _load_mnist_xy()
    dataset = TestTensorDataset(
        torch.tensor(x_batch).float(), torch.tensor(y_batch).long()
    )
    metadata = LabelGroupingDataset.metadata_cls(seed=27)
    mapping = ClassMapping(
        class_to_group=ClassMapping._generate(
            n_classes=10, n_groups=2, seed=27
        ),
        n_classes=10,
        n_groups=2,
        seed=27,
    )
    return LabelGroupingDataset(
        dataset,
        metadata=metadata,
        class_to_group=mapping.class_to_group,
        n_classes=mapping.n_classes,
        n_groups=mapping.n_groups,
    )


@pytest.fixture
def load_mislabeling_mnist_dataset():
    x_batch, y_batch = _load_mnist_xy()
    dataset = TestTensorDataset(
        torch.tensor(x_batch).float(), torch.tensor(y_batch).long()
    )
    metadata = LabelFlippingDataset.metadata_cls(p=1.0, seed=27)
    return LabelFlippingDataset(dataset, metadata=metadata)


@pytest.fixture
def load_mnist_dataloader():
    x_batch, y_batch = _load_mnist_xy()
    dataset = TensorDataset(
        torch.tensor(x_batch).float(), torch.tensor(y_batch).long()
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=MINI_BATCH_SIZE, shuffle=False
    )


@pytest.fixture
def load_mnist_test_samples_1():
    return torch.load("tests/assets/dataset/test_dataset.pt")


@pytest.fixture
def load_mnist_test_samples_batches():
    return [
        torch.load("tests/assets/dataset/test_dataset.pt"),
        torch.load("tests/assets/dataset/test_dataset_2.pt"),
    ]


@pytest.fixture
def load_mnist_test_labels_1():
    return torch.load("tests/assets/dataset/test_labels.pt")


@pytest.fixture
def load_mnist_test_labels_1_list():
    return torch.load("tests/assets/dataset/test_labels.pt").tolist()


@pytest.fixture
def load_mnist_explanations_similarity_1():
    return torch.load("tests/assets/tda/mnist_SimilarityInfluence_tda.pt")


@pytest.fixture
def load_mnist_explanations_dot_similarity_1():
    return torch.load("tests/assets/tda/mnist_SimilarityInfluence_dot_tda.pt")


@pytest.fixture
def load_mnist_dataset_explanations():
    return torch.rand((MINI_BATCH_SIZE, MINI_BATCH_SIZE))


@pytest.fixture
def load_fashion_mnist_to_mnist_transform():
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
        ]
    )


@pytest.fixture
def mnist_white_square_transformation():
    def add_white_square(img):
        img[:, 8:13, 10:15] = 1.0
        return img

    return add_white_square


@pytest.fixture
def load_subset_indices_lds():
    return "subset_indices.yaml"


@pytest.fixture
def load_pretrained_models_lds():
    return [
        "tests/assets/lds_checkpoints/model_subset_0.pt",
        "tests/assets/lds_checkpoints/model_subset_1.pt",
        "tests/assets/lds_checkpoints/model_subset_2.pt",
        "tests/assets/lds_checkpoints/model_subset_3.pt",
    ]


def _load_mnist_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@pytest.fixture
def load_mnist_unit_test_config():
    return _load_mnist_yaml(
        "tests/assets/mnist_local_bench/83edb41-default_ClassDetection.yaml"
    )


@pytest.fixture
def load_mnist_unit_test_config_one_cycle():
    config = _load_mnist_yaml(
        "tests/assets/mnist_local_bench/83edb41-default_ClassDetection.yaml"
    )
    config["model"]["trainer"]["scheduler"] = "one_cycle"
    config["model"]["trainer"]["scheduler_kwargs"] = {
        "max_lr": 0.02,
        "interval": "step",
    }
    return config


@pytest.fixture
def load_mnist_unit_test_config_num_ckpts_2():
    config = _load_mnist_yaml(
        "tests/assets/mnist_local_bench/83edb41-default_ClassDetection.yaml"
    )
    config["num_checkpoints"] = 2
    return config


@pytest.fixture
def load_mnist_unit_test_config_hf():
    return _load_mnist_yaml(config_map["mnist_class_detection_unit"])


@pytest.fixture
def load_mnist_linear_datamodeling_config(
    load_pretrained_models_lds, load_subset_indices_lds
):
    return _load_mnist_yaml(
        "tests/assets/mnist_local_bench/83edb41-default_LDS.yaml"
    )


@pytest.fixture
def load_mnist_mislabeling_config():
    return _load_mnist_yaml(
        "tests/assets/mnist_local_bench/"
        "83edb41-default_MislabelingDetection.yaml"
    )


@pytest.fixture
def load_mnist_subclass_config():
    return _load_mnist_yaml(
        "tests/assets/mnist_local_bench/83edb41-default_SubclassDetection.yaml"
    )


@pytest.fixture
def load_mnist_shortcut_config():
    return _load_mnist_yaml(
        "tests/assets/mnist_local_bench/83edb41-default_ShortcutDetection.yaml"
    )


@pytest.fixture
def load_mnist_mixed_config():
    return _load_mnist_yaml(
        "tests/assets/mnist_local_bench/83edb41-default_MixedDatasets.yaml"
    )


@pytest.fixture
def load_mnist_lds_config():
    return _load_mnist_yaml(
        "tests/assets/mnist_local_bench/83edb41-default_LDS.yaml"
    )
