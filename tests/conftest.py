import json
import os
import pickle

import numpy as np
import pytest
import torch
import torchvision
from torch.utils.data import TensorDataset

from quanda.benchmarks.downstream_eval import (
    ClassDetection,
    MislabelingDetection,
    ShortcutDetection,
    SubclassDetection,
)
from quanda.benchmarks.heuristics import (
    MixedDatasets,
    ModelRandomization,
    TopKCardinality,
)
from quanda.utils.datasets.transformed.label_flipping import (
    LabelFlippingDataset,
)
from quanda.utils.datasets.transformed.label_grouping import (
    LabelGroupingDataset,
)
from quanda.utils.training.base_pl_module import BasicLightningModule
from tests.models import LeNet

MNIST_IMAGE_SIZE = 28
BATCH_SIZE = 124
MINI_BATCH_SIZE = 8
RANDOM_SEED = 42


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "local: only run this test if running locally"
    )


def pytest_runtest_setup(item):
    if "local" in item.keywords and os.getenv("GITHUB_ACTIONS"):
        pytest.skip("Skipping local-only tests on GitHub Actions")


class TestTensorDataset(TensorDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = self.tensors[0]
        self._targets = self.tensors[1]

    def __getitem__(self, index):
        return self._data[index], self._targets[index].item()


@pytest.fixture
def load_dataset():
    x = torch.stack([torch.rand(2, 2), torch.rand(2, 2), torch.rand(2, 2)])
    y = torch.tensor([0, 1, 0]).long()
    return torch.utils.data.TensorDataset(x, y)


@pytest.fixture
def load_rand_tensor():
    return torch.rand(10, 10).float()


@pytest.fixture
def load_rand_test_predictions():
    return torch.randint(0, 10, (10000,))


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
        "tests/assets/mnist_test_suite_1/mnist_seed_27_poisoned_labels.json",
        "r",
    ) as f:
        return json.load(f)


@pytest.fixture
def load_mnist_checkpoints():
    """Get paths to five checkpoints from a single training run."""
    checkpoint_paths = [
        "tests/assets/mnist_checkpoints/checkpoint-00",
        "tests/assets/mnist_checkpoints/checkpoint-01",
        "tests/assets/mnist_checkpoints/checkpoint-02",
        "tests/assets/mnist_checkpoints/checkpoint-03",
        "tests/assets/mnist_checkpoints/checkpoint-04",
    ]
    return checkpoint_paths


@pytest.fixture
def load_mnist_model():
    """Load a pre-trained LeNet classification model (architecture at quantus/helpers/models)."""
    model = LeNet()
    model.load_state_dict(
        torch.load(
            "tests/assets/mnist", map_location="cpu", pickle_module=pickle
        )
    )
    return model


@pytest.fixture
def load_mnist_last_checkpoint():
    """Load the path to the last checkpoint of a pre-trained LeNet classification model."""
    return "tests/assets/mnist"


@pytest.fixture
def load_mnist_pl_module():
    """Load a pre-trained LeNet classification model (architecture at quantus/helpers/models)."""
    model = LeNet()
    model.load_state_dict(
        torch.load(
            "tests/assets/mnist", map_location="cpu", pickle_module=pickle
        )
    )

    pl_module = BasicLightningModule(
        model=model,
        optimizer=torch.optim.SGD,
        lr=0.01,
        criterion=torch.nn.CrossEntropyLoss(),
    )

    return pl_module


@pytest.fixture
def load_mnist_grouped_model():
    """Load a pre-trained LeNet classification model (architecture at quantus/helpers/models)."""
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
    """Load a not trained LeNet classification model (architecture at quantus/helpers/models)."""
    return LeNet()


@pytest.fixture
def load_mnist_dataset():
    """Load a batch of MNIST digits: inputs and outputs to use for testing."""
    x_batch = (
        np.loadtxt("tests/assets/mnist_x")
        .astype(float)
        .reshape((BATCH_SIZE, 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE))
    )[:MINI_BATCH_SIZE]
    y_batch = np.loadtxt("tests/assets/mnist_y").astype(int)[:MINI_BATCH_SIZE]
    dataset = TestTensorDataset(
        torch.tensor(x_batch).float(), torch.tensor(y_batch).long()
    )
    return dataset


@pytest.fixture
def load_mnist_labels():
    y_batch = np.loadtxt("tests/assets/mnist_y").astype(int)[:MINI_BATCH_SIZE]
    return torch.tensor(y_batch).long()


@pytest.fixture
def load_mnist_adversarial_indices():
    y_batch = np.loadtxt("tests/assets/mnist_y").astype(int)[:MINI_BATCH_SIZE]
    return [int(y == 1) for y in y_batch]


@pytest.fixture
def load_grouped_mnist_dataset():
    x_batch = (
        np.loadtxt("tests/assets/mnist_x")
        .astype(float)
        .reshape((BATCH_SIZE, 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE))
    )[:MINI_BATCH_SIZE]
    y_batch = np.loadtxt("tests/assets/mnist_y").astype(int)[:MINI_BATCH_SIZE]
    dataset = TestTensorDataset(
        torch.tensor(x_batch).float(), torch.tensor(y_batch).long()
    )
    return LabelGroupingDataset(
        dataset,
        n_classes=10,
        n_groups=2,
        class_to_group="random",
        seed=27,
    )


@pytest.fixture
def load_mislabeling_mnist_dataset():
    x_batch = (
        np.loadtxt("tests/assets/mnist_x")
        .astype(float)
        .reshape((BATCH_SIZE, 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE))
    )[:MINI_BATCH_SIZE]
    y_batch = np.loadtxt("tests/assets/mnist_y").astype(int)[:MINI_BATCH_SIZE]
    dataset = TestTensorDataset(
        torch.tensor(x_batch).float(), torch.tensor(y_batch).long()
    )
    return LabelFlippingDataset(
        dataset,
        n_classes=10,
        p=1.0,
        seed=27,
    )


@pytest.fixture
def load_mnist_dataloader():
    """Load a batch of MNIST digits: inputs and outputs to use for testing."""
    x_batch = (
        np.loadtxt("tests/assets/mnist_x")
        .astype(float)
        .reshape((BATCH_SIZE, 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE))
    )[:MINI_BATCH_SIZE]
    y_batch = np.loadtxt("tests/assets/mnist_y").astype(int)[:MINI_BATCH_SIZE]
    dataset = TensorDataset(
        torch.tensor(x_batch).float(), torch.tensor(y_batch).long()
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=MINI_BATCH_SIZE, shuffle=False
    )
    return dataloader


@pytest.fixture
def load_mnist_test_samples_1():
    return torch.load("tests/assets/mnist_test_suite_1/test_dataset.pt")


@pytest.fixture
def load_mnist_test_samples_batches():
    return [
        torch.load("tests/assets/mnist_test_suite_1/test_dataset.pt"),
        torch.load("tests/assets/mnist_test_suite_1/test_dataset_2.pt"),
    ]


@pytest.fixture
def load_mnist_test_labels_1():
    return torch.load("tests/assets/mnist_test_suite_1/test_labels.pt")


@pytest.fixture
def load_mnist_explanations_similarity_1():
    return torch.load(
        "tests/assets/mnist_test_suite_1/mnist_SimilarityInfluence_tda.pt"
    )


@pytest.fixture
def load_mnist_explanations_dot_similarity_1():
    return torch.load(
        "tests/assets/mnist_test_suite_1/mnist_SimilarityInfluence_dot_tda.pt"
    )


@pytest.fixture
def load_mnist_explanations_trak_1():
    return torch.load("tests/assets/mnist_test_suite_1/mnist_TRAK_tda.pt")


@pytest.fixture
def load_mnist_explanations_trak_si_1():
    return torch.load("tests/assets/mnist_test_suite_1/mnist_TRAK_tda_si.pt")


@pytest.fixture
def load_mnist_dataset_explanations():
    return torch.rand((MINI_BATCH_SIZE, MINI_BATCH_SIZE))


@pytest.fixture
def torch_cross_entropy_loss_object():
    return torch.nn.CrossEntropyLoss()


@pytest.fixture
def torch_constant_lr_scheduler_type():
    return torch.optim.lr_scheduler.ConstantLR


@pytest.fixture
def torch_sgd_optimizer():
    return torch.optim.SGD


@pytest.fixture
def load_fashion_mnist_to_mnist_transform():
    # change tp black and white

    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Grayscale(num_output_channels=1),
            torchvision.transforms.Resize((28, 28)),
            torchvision.transforms.ToTensor(),
        ]
    )


@pytest.fixture
def load_fashion_mnist_path():
    return "tests/assets/fashion_mnist_examples"


@pytest.fixture
def mnist_white_square_transformation():
    def add_white_square(img):
        img[:, 8:13, 10:15] = (
            1.0  # Paste it onto the image at the specified position
        )
        return img

    return add_white_square


@pytest.fixture(scope="session")
def mnist_class_detection_benchmark(tmp_path_factory):
    dst_eval = ClassDetection.download(
        name="mnist_class_detection",
        cache_dir=str(
            tmp_path_factory.mktemp("mnist_class_detection_benchmark")
        ),
        device="cpu",
    )
    return dst_eval


@pytest.fixture(scope="session")
def mnist_subclass_detection_benchmark(tmp_path_factory):
    dst_eval = SubclassDetection.download(
        name="mnist_subclass_detection",
        cache_dir=str(
            tmp_path_factory.mktemp("mnist_subclass_detection_benchmark")
        ),
        device="cpu",
    )
    return dst_eval


@pytest.fixture(scope="session")
def mnist_mislabeling_detection_benchmark(tmp_path_factory):
    dst_eval = MislabelingDetection.download(
        name="mnist_mislabeling_detection",
        cache_dir=str(
            tmp_path_factory.mktemp("mnist_mislabeling_detection_benchmark")
        ),
        device="cpu",
    )
    return dst_eval


@pytest.fixture(scope="session")
def mnist_shortcut_detection_benchmark(tmp_path_factory):
    dst_eval = ShortcutDetection.download(
        name="mnist_shortcut_detection",
        cache_dir=str(
            tmp_path_factory.mktemp("mnist_shortcut_detection_benchmark")
        ),
        device="cpu",
    )
    return dst_eval


@pytest.fixture(scope="session")
def mnist_mixed_datasets_benchmark(tmp_path_factory):
    dst_eval = MixedDatasets.download(
        name="mnist_mixed_datasets",
        cache_dir=str(
            tmp_path_factory.mktemp("mnist_mixed_datasets_benchmark")
        ),
        device="cpu",
    )
    return dst_eval


@pytest.fixture(scope="session")
def mnist_model_randomization_benchmark(tmp_path_factory):
    dst_eval = ModelRandomization.download(
        name="mnist_class_detection",
        cache_dir=str(
            tmp_path_factory.mktemp("mnist_class_detection_benchmark")
        ),
        device="cpu",
    )
    return dst_eval


@pytest.fixture(scope="session")
def mnist_top_k_cardinality_benchmark(tmp_path_factory):
    dst_eval = TopKCardinality.download(
        name="mnist_class_detection",
        cache_dir=str(
            tmp_path_factory.mktemp("mnist_class_detection_benchmark")
        ),
        device="cpu",
    )
    return dst_eval


@pytest.fixture
def get_lds_score():
    with open("tests/assets/lds_score.json", "r") as f:
        score_data = json.load(f)
    return score_data["lds_score"]
