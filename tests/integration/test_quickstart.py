import pytest
from torch.utils.data import random_split

# START1
from torch.utils.data import DataLoader
from tqdm import tqdm

from quanda.explainers.wrappers import CaptumSimilarity
from quanda.metrics.heuristics import ModelRandomizationMetric
# END1

# START5
from quanda.explainers.wrappers import CaptumSimilarity
from quanda.benchmarks.downstream_eval import SubclassDetection
# END5

# START8
from quanda.explainers.wrappers import CaptumSimilarity
from quanda.benchmarks.heuristics import TopKCardinality
# END8

# START11
import torch

from quanda.explainers.wrappers import CaptumSimilarity
from quanda.benchmarks.downstream_eval import MislabelingDetection
# END11

# START13_1
from quanda.utils.training.trainer import Trainer
# END13_1


@pytest.mark.explainers
@pytest.mark.parametrize(
    "test_id, model, dataset, batch_size",
    [
        (
            "quickstart",
            "load_mnist_model",
            "load_mnist_dataset",
            4,
        )
    ],
)
def test_quickstart(
    test_id,
    model,
    dataset,
    batch_size,
    request,
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)

    n_classes = 10
    train_set, eval_set = random_split(dataset, [6, 2])

    # START2
    DEVICE = "cpu"
    model.to(DEVICE)
    explainer_kwargs = {
        "layers": "fc_2",
        "model_id": "default_model_id",
        "cache_dir": "./cache"
    }
    explainer = CaptumSimilarity(
        model=model,
        train_dataset=train_set,
        **explainer_kwargs
    )
    # END2

    # START3
    explainer_kwargs = {
        "layers": "fc_2",
        "model_id": "randomized_model_id",
        "cache_dir": "./cache"
    }
    model_rand = ModelRandomizationMetric(
        model=model,
        model_id="randomized_model_id",
        cache_dir="./cache",
        train_dataset=train_set,
        explainer_cls=CaptumSimilarity,
        expl_kwargs=explainer_kwargs,
        correlation_fn="spearman",
        seed=42,
    )
    # END3

    # START4
    test_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)
    for test_tensor, _ in tqdm(test_loader):
        test_tensor = test_tensor.to(DEVICE)
        target = model(test_tensor).argmax(dim=-1)
        tda = explainer.explain(
            test_tensor=test_tensor,
            targets=target
        )
        model_rand.update(test_data=test_tensor,
                          explanations=tda, explanation_targets=target)

    print("Randomization metric output:", model_rand.compute())
    # END4

    # START6
    DEVICE = "cpu"
    model.to(DEVICE)

    explainer_kwargs = {
        "layers": "model.fc_2",
        "model_id": "default_model_id",
        "cache_dir": "./cache",
    }
    # END6

    # START7_1
    subclass_detect = SubclassDetection.download(
        name="mnist_subclass_detection",
        cache_dir="./cache",
        device="cpu",
    )
    # END7_1

    subclass_detect.base_dataset = torch.utils.data.Subset(
        subclass_detect.base_dataset, list(range(4))
    )
    subclass_detect.grouped_dataset = torch.utils.data.Subset(
        subclass_detect.grouped_dataset, list(range(4))
    )
    subclass_detect.eval_dataset = torch.utils.data.Subset(
        subclass_detect.eval_dataset, list(range(4))
    )

    # START7_2
    score = subclass_detect.evaluate(
        explainer_cls=CaptumSimilarity,
        expl_kwargs=explainer_kwargs,
        batch_size=batch_size,
    )["score"]
    print(f"Subclass Detection Score: {score}")
    # END7_2

    # START9
    DEVICE = "cpu"
    model.to(DEVICE)

    explainer_kwargs = {
        "layers": "fc_2",
        "model_id": "default_model_id",
        "cache_dir": "./cache"
    }
    # END9

    # START10
    topk_cardinality = TopKCardinality.assemble(
        model=model,
        train_dataset=train_set,
        eval_dataset=eval_set,
    )
    score = topk_cardinality.evaluate(
        explainer_cls=CaptumSimilarity,
        expl_kwargs=explainer_kwargs,
        batch_size=batch_size,
    )["score"]
    print(f"Top K Cardinality Score: {score}")
    # END10

    # START12
    DEVICE = "cpu"
    model.to(DEVICE)

    explainer_kwargs = {
        "layers": "fc_2",
        "model_id": "default_model_id",
        "cache_dir": "./cache"
    }
    # END12

    # START13_2
    trainer = Trainer(
        max_epochs=100,
        optimizer=torch.optim.SGD,
        lr=0.01,
        criterion=torch.nn.CrossEntropyLoss(),
    )
    # END13_2

    # START14
    mislabeling_detection = MislabelingDetection.generate(
        model=model,
        cache_dir="./cache",
        base_dataset=train_set,
        n_classes=n_classes,
        trainer=trainer,
    )

    score = mislabeling_detection.evaluate(
        explainer_cls=CaptumSimilarity,
        expl_kwargs=explainer_kwargs,
        batch_size=batch_size,
    )["score"]
    print(f"Mislabeling Detection Score: {score}")
    # END14
