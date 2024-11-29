import pytest
import torch.nn as nn
from datasets import load_dataset
from kronfluence.arguments import (  # type: ignore
    FactorArguments,
    ScoreArguments,
)
from kronfluence.utils.dataset import DataLoaderKwargs  # type: ignore
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from quanda.explainers.wrappers import (
    Kronfluence,
    kronfluence_explain,
    kronfluence_self_influence,
)


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, test_tensor, test_labels, task",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            "classification_task",
        ),
    ],
)
def test_kronfluence_explain(
    test_id,
    model,
    dataset,
    test_tensor,
    test_labels,
    task,
    request,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    test_labels = request.getfixturevalue(test_labels)
    task = request.getfixturevalue(task)

    explainer = Kronfluence(
        model=model,
        task=task,
        train_dataset=train_dataset,
        batch_size=1,
        device="cpu",
    )
    explanations = explainer.explain(
        test_tensor=test_tensor, targets=test_labels
    )

    assert explanations.shape == (
        len(test_tensor),
        len(train_dataset),
    ), "Training data attributions have incorrect shape"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, task",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "classification_task",
        ),
    ],
)
def test_kronfluence_self_influence(
    test_id,
    model,
    dataset,
    task,
    request,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    task = request.getfixturevalue(task)

    explainer = Kronfluence(
        model=model,
        task=task,
        train_dataset=train_dataset,
        batch_size=1,
        device="cpu",
    )
    self_influence_scores = explainer.self_influence()

    assert self_influence_scores.shape == (
        len(train_dataset),
    ), "Self-influence scores have incorrect shape"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, test_tensor, test_labels, task",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            "classification_task",
        ),
    ],
)
def test_kronfluence_explain_functional(
    test_id,
    model,
    dataset,
    test_tensor,
    test_labels,
    task,
    request,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    test_labels = request.getfixturevalue(test_labels)
    task = request.getfixturevalue(task)

    explanations = kronfluence_explain(
        model=model,
        task=task,
        test_tensor=test_tensor,
        explanation_targets=test_labels,
        train_dataset=train_dataset,
        batch_size=1,
        device="cpu",
    )

    assert explanations.shape == (
        len(test_tensor),
        len(train_dataset),
    ), "Training data attributions have incorrect shape"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, task",
    [
        (
            "mnist",
            "load_mnist_model",
            "load_mnist_dataset",
            "classification_task",
        ),
    ],
)
def test_kronfluence_self_influence_functional(
    test_id,
    model,
    dataset,
    task,
    request,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    task = request.getfixturevalue(task)

    self_influence_scores = kronfluence_self_influence(
        model=model,
        task=task,
        train_dataset=train_dataset,
        batch_size=1,
        device="cpu",
    )

    assert self_influence_scores.shape == (
        len(train_dataset),
    ), "Self-influence scores have incorrect shape"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, test_tensor, test_labels, task, factor_args, score_args, dataloader_kwargs",
    [
        (
            "mnist_optional",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            "classification_task",
            FactorArguments(strategy="identity"),
            ScoreArguments(damping_factor=1e-5),
            DataLoaderKwargs(num_workers=0),
        ),
    ],
)
def test_kronfluence_explain_with_optional_args(
    test_id,
    model,
    dataset,
    test_tensor,
    test_labels,
    task,
    factor_args,
    score_args,
    dataloader_kwargs,
    request,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    test_labels = request.getfixturevalue(test_labels)
    task = request.getfixturevalue(task)

    explainer = Kronfluence(
        model=model,
        task=task,
        train_dataset=train_dataset,
        batch_size=1,
        device="cpu",
        factor_args=factor_args,
        dataloader_kwargs=dataloader_kwargs,
        overwrite_output_dir=True,
    )

    explanations = explainer.explain(
        test_tensor=test_tensor,
        targets=test_labels,
        score_args=score_args,
        overwrite_output_dir=True,
    )

    assert explanations.shape == (
        len(test_tensor),
        len(train_dataset),
    ), "Training data attributions have incorrect shape"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, task, factor_args, score_args, dataloader_kwargs",
    [
        (
            "mnist_optional",
            "load_mnist_model",
            "load_mnist_dataset",
            "classification_task",
            FactorArguments(strategy="kfac"),
            ScoreArguments(damping_factor=1e-7),
            DataLoaderKwargs(num_workers=0),
        ),
    ],
)
def test_kronfluence_self_influence_with_optional_args(
    test_id,
    model,
    dataset,
    task,
    factor_args,
    score_args,
    dataloader_kwargs,
    request,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    task = request.getfixturevalue(task)

    explainer = Kronfluence(
        model=model,
        task=task,
        train_dataset=train_dataset,
        batch_size=1,
        device="cpu",
        factor_args=factor_args,
        dataloader_kwargs=dataloader_kwargs,
        overwrite_output_dir=True,
    )

    self_influence_scores = explainer.self_influence(
        score_args=score_args, overwrite_output_dir=True
    )

    assert self_influence_scores.shape == (
        len(train_dataset),
    ), "Self-influence scores have incorrect shape"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, test_tensor, test_labels, task, factor_args, score_args",
    [
        (
            "mnist_optional_functional",
            "load_mnist_model",
            "load_mnist_dataset",
            "load_mnist_test_samples_1",
            "load_mnist_test_labels_1",
            "classification_task",
            FactorArguments(strategy="identity"),
            ScoreArguments(damping_factor=1e-6),
        ),
    ],
)
def test_kronfluence_explain_functional_with_optional_args(
    test_id,
    model,
    dataset,
    test_tensor,
    test_labels,
    task,
    factor_args,
    score_args,
    request,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    test_tensor = request.getfixturevalue(test_tensor)
    test_labels = request.getfixturevalue(test_labels)
    task = request.getfixturevalue(task)

    explanations = kronfluence_explain(
        model=model,
        task=task,
        test_tensor=test_tensor,
        explanation_targets=test_labels,
        train_dataset=train_dataset,
        batch_size=1,
        device="cpu",
        factor_args=factor_args,
        score_args=score_args,
    )

    assert explanations.shape == (
        len(test_tensor),
        len(train_dataset),
    ), "Training data attributions have incorrect shape"


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, task, factor_args, score_args",
    [
        (
            "mnist_optional_functional",
            "load_mnist_model",
            "load_mnist_dataset",
            "classification_task",
            FactorArguments(strategy="kfac"),
            ScoreArguments(damping_factor=1e-4),
        ),
    ],
)
def test_kronfluence_self_influence_functional_with_optional_args(
    test_id,
    model,
    dataset,
    task,
    factor_args,
    score_args,
    request,
):
    model = request.getfixturevalue(model)
    train_dataset = request.getfixturevalue(dataset)
    task = request.getfixturevalue(task)

    self_influence_scores = kronfluence_self_influence(
        model=model,
        task=task,
        train_dataset=train_dataset,
        batch_size=1,
        device="cpu",
        factor_args=factor_args,
        score_args=score_args,
    )

    assert self_influence_scores.shape == (
        len(train_dataset),
    ), "Self-influence scores have incorrect shape"


# Partially copied from https://github.com/MadryLab/trak/blob/main/examples/qnli.py
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

TRAIN_SET_SIZE = 10
VAL_SET_SIZE = 4


class SequenceClassificationModel(nn.Module):
    """
    Wrapper for HuggingFace sequence classification models.
    """

    def __init__(self):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            "gchhablani/bert-base-cased-finetuned-qnli",
            num_labels=2,
            finetuning_task="qnli",
            cache_dir=None,
            revision="main",
            token=None,
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            "gchhablani/bert-base-cased-finetuned-qnli",
            config=self.config,
            cache_dir=None,
            revision="main",
            token=None,
            ignore_mismatched_sizes=False,
        )

        self.model.eval()

    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )


def get_dataset(split, inds=None):
    raw_datasets = load_dataset(
        "glue",
        "qnli",
        cache_dir=None,
        use_auth_token=None,
    )
    sentence1_key, sentence2_key = GLUE_TASK_TO_KEYS["qnli"]

    tokenizer = AutoTokenizer.from_pretrained(
        "gchhablani/bert-base-cased-finetuned-qnli",
        cache_dir=None,
        use_fast=True,
        revision="main",
        token=None,
    )

    padding = "max_length"
    max_seq_length = 128

    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(
            *args, padding=padding, max_length=max_seq_length, truncation=True
        )

        result["labels"] = examples["label"]

        return result

    raw_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=(not False),
        desc="Running tokenizer on dataset",
    )

    if split == "train":
        train_dataset = raw_datasets["train"]
        ds = train_dataset
    else:
        eval_dataset = raw_datasets["validation"]
        ds = eval_dataset
    return ds


def init_model(ckpt_path, device="cpu"):
    model = SequenceClassificationModel()
    return model


def init_loaders(batch_size=2):
    ds_train = get_dataset("train")
    ds_train = ds_train.select(range(TRAIN_SET_SIZE))
    ds_val = get_dataset("validation")
    ds_val = ds_val.select(range(VAL_SET_SIZE))

    tokenizer = AutoTokenizer.from_pretrained(
        "gchhablani/bert-base-cased-finetuned-qnli",
        cache_dir=None,
        use_fast=True,
        revision="main",
        token=None,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    return DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    ), DataLoader(
        ds_val, batch_size=batch_size, shuffle=False, collate_fn=data_collator
    )


def process_batch(batch):
    return (
        batch["input_ids"],
        batch["token_type_ids"],
        batch["attention_mask"],
        batch["labels"],
    )


@pytest.mark.explainers
def test_kronfluence_self_influence_qnli(
    load_qnli_model,
    text_classification_task,
):
    loader_train, loader_val = init_loaders()

    model = init_model(".", "cpu")

    train_dataset = loader_train.dataset
    test_dataset = loader_val.dataset

    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "labels"],
    )
    test_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "token_type_ids", "labels"],
    )

    explainer = Kronfluence(
        model=model,
        task=text_classification_task,
        train_dataset=train_dataset,
        batch_size=1,
        device="cpu",
    )
    self_influence_scores = explainer.self_influence()

    assert self_influence_scores.shape == (
        len(train_dataset),
    ), "Self-influence scores have incorrect shape"
