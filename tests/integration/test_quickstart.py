import os
import pytest
import yaml
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


@pytest.mark.skipif("GITHUB_ACTIONS" in os.environ, reason="Skip on GitHub Actions")
@pytest.mark.benchmarks
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

    eval_dataset = dataset


    # START2
    DEVICE = "cpu"
    model.to(DEVICE)
    cache_dir = "quanda_benchmark_quickstart_cache"

    explainer_kwargs = {
        "layers": "fc_2",
        "model_id": "default_model_id",
        "cache_dir": cache_dir
    }
    explainer = CaptumSimilarity(
        model=model,
        train_dataset=dataset,
        **explainer_kwargs
    )
    # END2

    # START3
    explainer_kwargs = {
        "layers": "fc_2",
        "model_id": "randomized_model_id",
        "cache_dir": cache_dir
    }
    model_rand = ModelRandomizationMetric(
        model=model,
        model_id="randomized_model_id",
        cache_dir=cache_dir,
        train_dataset=dataset,
        explainer_cls=CaptumSimilarity,
        expl_kwargs=explainer_kwargs,
        correlation_fn="spearman",
        seed=42,
    )
    # END3

    # START4
    test_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)
    for test_data, _ in tqdm(test_loader):
        test_data = test_data.to(DEVICE)
        target = model(test_data).argmax(dim=-1)
        tda = explainer.explain(
            test_data=test_data,
            targets=target
        )
        model_rand.update(explanations=tda,
                          test_data=test_data, test_targets=target)

    print("Randomization metric output:", model_rand.compute())
    # END4

    # START6
    DEVICE = "cpu"
    model.to(DEVICE)

    explainer_kwargs = {
        "layers": "fc_2",
        "model_id": "default_model_id",
        "cache_dir": cache_dir,
    }
    # END6

    # START7_1
    subclass_detect = SubclassDetection.load_pretrained(
        bench_id="mnist_subclass_detection",
        cache_dir=cache_dir,
    )
    # END7_1

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
        "cache_dir": cache_dir
    }
    # END9

    # START10
    with open(
        "tests/assets/mnist_test_suite_2/7ed30b3-default_TopKCardinality.yaml",
        "r",
    ) as f:
        top_k_config = yaml.safe_load(f)

    topk_cardinality = TopKCardinality.from_config(
        top_k_config,

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
        "model_id": "top_k_model",
        "cache_dir": cache_dir
    }
    # END12

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
