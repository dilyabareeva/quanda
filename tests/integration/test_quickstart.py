import os

import pytest

# END8
# START11
import yaml

# START1
from torch.utils.data import DataLoader
from tqdm import tqdm

# END1
# START5
from quanda.benchmarks.downstream_eval import (
    MislabelingDetection,
    SubclassDetection,
)

# END5
# START8
from quanda.benchmarks.heuristics import TopKCardinality
from quanda.explainers.wrappers import CaptumSimilarity
from quanda.metrics.heuristics import ModelRandomizationMetric

# END11

# START13_1
# END13_1


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ, reason="Skip on GitHub Actions"
)
@pytest.mark.integration
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
    tmp_path,
    request,
):
    model = request.getfixturevalue(model)
    dataset = request.getfixturevalue(dataset)

    eval_set = dataset

    cache_dir = str(
        os.path.join(tmp_path, "quanda_benchmark_quickstart_cache")
    )

    # START2
    DEVICE = "cpu"
    model.to(DEVICE)

    explainer_kwargs = {
        "layers": "fc_2",
        "model_id": "default_model_id",
        "cache_dir": cache_dir,
    }
    explainer = CaptumSimilarity(
        model=model, train_dataset=dataset, **explainer_kwargs
    )
    # END2

    # START3
    explainer_kwargs = {
        "layers": "fc_2",
        "model_id": "randomized_model_id",
        "cache_dir": cache_dir,
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
        tda = explainer.explain(test_data=test_data, targets=target)
        model_rand.update(
            explanations=tda, test_data=test_data, test_targets=target
        )

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
        "cache_dir": cache_dir,
    }
    # END9

    # START10
    with open(
        "tests/assets/mnist_local_bench/124bfe7-default_TopKCardinality.yaml",
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
        "cache_dir": cache_dir,
    }
    # END12

    # START14
    with open(
        "tests/assets/mnist_local_bench/124bfe7-default_MislabelingDetection.yaml",
        "r",
    ) as f:
        mislabel_config = yaml.safe_load(f)

    mislabeling_detection = MislabelingDetection.train(
        mislabel_config,
        device="cpu",
    )
    score = mislabeling_detection.evaluate(
        explainer_cls=CaptumSimilarity,
        expl_kwargs=explainer_kwargs,
        batch_size=batch_size,
    )["score"]
    print(f"Mislabeling Detection Score: {score}")
    # END14
