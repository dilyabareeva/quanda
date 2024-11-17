import json
import os
import pickle
from typing import Dict, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from kronfluence.task import Task  # type: ignore
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

# Copied from https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py.
GLUE_TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


MNIST_IMAGE_SIZE = 28
BATCH_SIZE = 124
MINI_BATCH_SIZE = 8
RANDOM_SEED = 42


def pytest_configure(config):
    config.addinivalue_line("markers", "local: only run this test if running locally")


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
        [[i * 1.0 for i in range(8)], [i * 1.0 for i in range(8)], [i * 1.0 for i in range(8)]], dtype=torch.float
    )


@pytest.fixture
def range_ranking():
    return torch.tensor([i for i in range(8)])


@pytest.fixture
def mnist_seed_27_mislabeling_labels():
    with open("tests/assets/mnist_test_suite_1/mnist_seed_27_poisoned_labels.json", "r") as f:
        return json.load(f)


@pytest.fixture
def get_mnist_checkpoints():
    """Get paths to five checkpoints from a single training run."""
    checkpoint_paths = [
        "tests/assets/mnist_checkpoints/checkpoint-00",
        "tests/assets/mnist_checkpoints/checkpoint-01",
        "tests/assets/mnist_checkpoints/checkpoint-02",
        "tests/assets/mnist_checkpoints/checkpoint-03",
        "tests/assets/mnist_checkpoints/checkpoint-04",
    ]
    checkpoints = []
    for path in checkpoint_paths:
        checkpoints.append(path)
    return checkpoints


@pytest.fixture
def load_mnist_model():
    """Load a pre-trained LeNet classification model (architecture at quantus/helpers/models)."""
    model = LeNet()
    model.load_state_dict(torch.load("tests/assets/mnist", map_location="cpu", pickle_module=pickle))
    return model


@pytest.fixture
def load_mnist_pl_module():
    """Load a pre-trained LeNet classification model (architecture at quantus/helpers/models)."""
    model = LeNet()
    model.load_state_dict(torch.load("tests/assets/mnist", map_location="cpu", pickle_module=pickle))

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
    model.load_state_dict(torch.load("tests/assets/mnist_grouped_model", map_location="cpu", pickle_module=pickle))
    return model


@pytest.fixture
def load_init_mnist_model():
    """Load a not trained LeNet classification model (architecture at quantus/helpers/models)."""
    return LeNet()


@pytest.fixture
def load_mnist_dataset():
    """Load a batch of MNIST digits: inputs and outputs to use for testing."""
    x_batch = (np.loadtxt("tests/assets/mnist_x").astype(float).reshape((BATCH_SIZE, 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)))[
        :MINI_BATCH_SIZE
    ]
    y_batch = np.loadtxt("tests/assets/mnist_y").astype(int)[:MINI_BATCH_SIZE]
    dataset = TestTensorDataset(torch.tensor(x_batch).float(), torch.tensor(y_batch).long())
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
    x_batch = (np.loadtxt("tests/assets/mnist_x").astype(float).reshape((BATCH_SIZE, 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)))[
        :MINI_BATCH_SIZE
    ]
    y_batch = np.loadtxt("tests/assets/mnist_y").astype(int)[:MINI_BATCH_SIZE]
    dataset = TestTensorDataset(torch.tensor(x_batch).float(), torch.tensor(y_batch).long())
    return LabelGroupingDataset(
        dataset,
        n_classes=10,
        n_groups=2,
        class_to_group="random",
        seed=27,
    )


@pytest.fixture
def load_mislabeling_mnist_dataset():
    x_batch = (np.loadtxt("tests/assets/mnist_x").astype(float).reshape((BATCH_SIZE, 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)))[
        :MINI_BATCH_SIZE
    ]
    y_batch = np.loadtxt("tests/assets/mnist_y").astype(int)[:MINI_BATCH_SIZE]
    dataset = TestTensorDataset(torch.tensor(x_batch).float(), torch.tensor(y_batch).long())
    return LabelFlippingDataset(
        dataset,
        n_classes=10,
        p=1.0,
        seed=27,
    )


@pytest.fixture
def load_mnist_dataloader():
    """Load a batch of MNIST digits: inputs and outputs to use for testing."""
    x_batch = (np.loadtxt("tests/assets/mnist_x").astype(float).reshape((BATCH_SIZE, 1, MNIST_IMAGE_SIZE, MNIST_IMAGE_SIZE)))[
        :MINI_BATCH_SIZE
    ]
    y_batch = np.loadtxt("tests/assets/mnist_y").astype(int)[:MINI_BATCH_SIZE]
    dataset = TensorDataset(torch.tensor(x_batch).float(), torch.tensor(y_batch).long())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=MINI_BATCH_SIZE, shuffle=False)
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
    return torch.load("tests/assets/mnist_test_suite_1/mnist_SimilarityInfluence_tda.pt")


@pytest.fixture
def load_mnist_explanations_dot_similarity_1():
    return torch.load("tests/assets/mnist_test_suite_1/mnist_SimilarityInfluence_dot_tda.pt")


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
        img[:, 8:13, 10:15] = 1.0  # Paste it onto the image at the specified position
        return img

    return add_white_square


@pytest.fixture(scope="session")
def mnist_class_detection_benchmark(tmp_path_factory):
    dst_eval = ClassDetection.download(
        name="mnist_class_detection",
        cache_dir=str(tmp_path_factory.mktemp("mnist_class_detection_benchmark")),
        device="cpu",
    )
    return dst_eval


@pytest.fixture(scope="session")
def mnist_subclass_detection_benchmark(tmp_path_factory):
    dst_eval = SubclassDetection.download(
        name="mnist_subclass_detection",
        cache_dir=str(tmp_path_factory.mktemp("mnist_subclass_detection_benchmark")),
        device="cpu",
    )
    return dst_eval


@pytest.fixture(scope="session")
def mnist_mislabeling_detection_benchmark(tmp_path_factory):
    dst_eval = MislabelingDetection.download(
        name="mnist_mislabeling_detection",
        cache_dir=str(tmp_path_factory.mktemp("mnist_mislabeling_detection_benchmark")),
        device="cpu",
    )
    return dst_eval


@pytest.fixture(scope="session")
def mnist_shortcut_detection_benchmark(tmp_path_factory):
    dst_eval = ShortcutDetection.download(
        name="mnist_shortcut_detection",
        cache_dir=str(tmp_path_factory.mktemp("mnist_shortcut_detection_benchmark")),
        device="cpu",
    )
    return dst_eval


@pytest.fixture(scope="session")
def mnist_mixed_datasets_benchmark(tmp_path_factory):
    dst_eval = MixedDatasets.download(
        name="mnist_mixed_datasets",
        cache_dir=str(tmp_path_factory.mktemp("mnist_mixed_datasets_benchmark")),
        device="cpu",
    )
    return dst_eval


@pytest.fixture(scope="session")
def mnist_model_randomization_benchmark(tmp_path_factory):
    dst_eval = ModelRandomization.download(
        name="mnist_class_detection",
        cache_dir=str(tmp_path_factory.mktemp("mnist_class_detection_benchmark")),
        device="cpu",
    )
    return dst_eval


@pytest.fixture(scope="session")
def mnist_top_k_cardinality_benchmark(tmp_path_factory):
    dst_eval = TopKCardinality.download(
        name="mnist_class_detection",
        cache_dir=str(tmp_path_factory.mktemp("mnist_class_detection_benchmark")),
        device="cpu",
    )
    return dst_eval


@pytest.fixture
def get_lds_score():
    with open("tests/assets/lds_score.json", "r") as f:
        score_data = json.load(f)
    return score_data["lds_score"]


class ClassificationTask(Task):
    # Copied from: https://github.com/pomonam/kronfluence/blob/main/examples/cifar/analyze.py
    def compute_train_loss(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        inputs, labels = batch
        logits = model(inputs)
        if not sample:
            return F.cross_entropy(logits, labels, reduction="sum")
        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
            sampled_labels = torch.multinomial(
                probs,
                num_samples=1,
            ).flatten()
        return F.cross_entropy(logits, sampled_labels, reduction="sum")

    def compute_measurement(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        model: nn.Module,
    ) -> torch.Tensor:
        # Copied from: https://github.com/MadryLab/trak/blob/main/trak/modelout_functions.py.
        inputs, labels = batch
        logits = model(inputs)

        bindex = torch.arange(logits.shape[0]).to(device=logits.device, non_blocking=False)
        logits_correct = logits[bindex, labels]

        cloned_logits = logits.clone()
        cloned_logits[bindex, labels] = torch.tensor(-torch.inf, device=logits.device, dtype=logits.dtype)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return -margins.sum()


@pytest.fixture
def classification_task():
    return ClassificationTask()


class TextClassificationTask(Task):
    def compute_train_loss(
        self,
        batch: Dict[str, torch.Tensor],
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        ).logits

        if not sample:
            return F.cross_entropy(logits, batch["labels"], reduction="sum")
        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits.detach(), dim=-1)
            sampled_labels = torch.multinomial(
                probs,
                num_samples=1,
            ).flatten()
        return F.cross_entropy(logits, sampled_labels, reduction="sum")

    def compute_measurement(
        self,
        batch: Dict[str, torch.Tensor],
        model: nn.Module,
    ) -> torch.Tensor:
        # Copied from: https://github.com/MadryLab/trak/blob/main/trak/modelout_functions.py.
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
        ).logits

        labels = batch["labels"]
        bindex = torch.arange(logits.shape[0]).to(device=logits.device, non_blocking=False)
        logits_correct = logits[bindex, labels]

        cloned_logits = logits.clone()
        cloned_logits[bindex, labels] = torch.tensor(-torch.inf, device=logits.device, dtype=logits.dtype)

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        return -margins.sum()

    def get_attention_mask(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return batch["attention_mask"]


@pytest.fixture
def text_classification_task():
    return TextClassificationTask()
