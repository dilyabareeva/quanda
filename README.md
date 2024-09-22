<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="/assets/readme/quanda_panda_black_bg.png">
  <source media="(prefers-color-scheme: light)" srcset="/assets/readme/quanda_panda_no_bg.png">
  <img width="500" alt="quanda" src="/assets/readme/quanda_panda_black_bg.png">
</picture>
</p>

<p align="center">
  Toolkit for <b>quan</b>titative evaluation of <b>d</b>ata <b>a</b>ttribution methods in <b>PyTorch</b>.
</p>


![py_versions](https://github-production-user-asset-6210df.s3.amazonaws.com/44092813/345210448-36499a1d-aefb-455f-b73a-57ca4794f31f.svg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20240904%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240904T071921Z&X-Amz-Expires=300&X-Amz-Signature=44ff9964c41d4ca7cc9a636178647e58e46e9b12ad4c213366aa2db149a21044&X-Amz-SignedHeaders=host&actor_id=44092813&key_id=0&repo_id=777729549)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![mypy](https://img.shields.io/badge/mypy-checked-green)
![flake8](https://img.shields.io/badge/flake8-checked-blueviolet)

**quanda** _is currently under active development so carefully note the release version to ensure reproducibility of your work._


## 🐼 Library overview
**Training data attribution** (TDA) methods attribute model output to its training samples ([Koh and Liang, 2017](https://proceedings.mlr.press/v70/koh17a.html); [Yeh et al., 2018](https://proceedings.neurips.cc/paper/2018/hash/8a7129b8f3edd95b7d969dfc2c8e9d9d-Abstract.html); [Park et al., 2023](https://proceedings.mlr.press/v202/park23c.html); [Pruthi et al., 2020](https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html); [Bae et al., 2024](https://arxiv.org/abs/2405.12186)). Outside of being used for understanding models, TDA has also found usage in a large variety of applications such as debugging model behavior ([Koh and Liang, 2017](https://proceedings.mlr.press/v70/koh17a.html); [Yeh et al., 2018](https://proceedings.neurips.cc/paper/2018/hash/8a7129b8f3edd95b7d969dfc2c8e9d9d-Abstract.html); [K and Søgaard, 2021](https://arxiv.org/abs/2111.04683); [Guo et al., 2021](https://aclanthology.org/2021.emnlp-main.808)), data summarization ([Khanna et al., 2019](https://proceedings.mlr.press/v89/khanna19a.html); [Marion et al., 2023](https://openreview.net/forum?id=XUIYn3jo5T); [Yang et al., 2023](https://openreview.net/forum?id=4wZiAXD29TQ)), dataset selection ([Engstrom et al., 2024](https://openreview.net/forum?id=GC8HkKeH8s); [Chhabra et al., 2024](https://openreview.net/forum?id=HE9eUQlAvo)), fact tracing ([Akyurek et al., 2022](https://aclanthology.org/2022.findings-emnlp.180)) and machine unlearning ([Warnecke
et al., 2023](https://arxiv.org/abs/2108.11577)).

The evaluation of TDA methods is a difficult task, especially due to the computationally demanding and noisy nature of the ground truths. ([Basu et al.](https://arxiv.org/abs/2006.14651); [Nguyen et al.](https://arxiv.org/abs/2305.19765)). For this reason, the community has proposed various sanity checks ([Hanawa et al., 2021](https://openreview.net/forum?id=9uvhpyQwzM_)) and downstream tasks ([Koh and Liang, 2017](https://proceedings.mlr.press/v70/koh17a.html);[Khanna et al., 2019](https://proceedings.mlr.press/v89/khanna19a.html);[Karthikeyan et al.](https://arxiv.org/abs/2111.04683)) to evaluate the effectiveness of TDA methods. 

### Library Features

- **Unified TDA Interface**: Quanda provides a unified interface for various TDA methods, allowing users to easily switch between different methods.
- **Metrics**: Quanda provides a set of metrics to evaluate the effectiveness of TDA methods. These metrics are based on the latest research in the field.
- **Benchmarking**: Quanda provides a benchmarking tool to evaluate the performance of TDA methods on a given model, dataset and problem. As many TDA evaluation methods require access to ground truth, our benchmarking tools allow to generate a controlled setting with ground truth, and then compare the performance of different TDA methods on this setting.

### Supported TDA Methods

TODO: table with links to original implementations

### Metrics

- **Identical Class / Identical Subclass** ([Hanawa et al., 2021](https://openreview.net/forum?id=9uvhpyQwzM_)): Measures the proportion of identical classes or subclasses in the top-1 training samples over the test dataset. If the attributions are based on similarity, they are expected to be predictive of the class of the test datapoint, as well as different subclasses under a single label.

-  **Top-K Overlap**  ([Barshan et al., 2020](http://proceedings.mlr.press/v108/barshan20a/barshan20a.pdf)): Measures the cardinality of the union of the top-K training samples. Since the attributions are expected to be dependent on the test input, they are expected to vary heavily for different test points, resulting in a low overlap (high metric value).

- **Model Randomization** ([Hanawa et al., 2021](https://openreview.net/forum?id=9uvhpyQwzM_)): Measures the correlation between the original TDA and the TDA of a model with randomized weights. Since the attributions are expected to depend on model parameters, the correlation between original and randomized attributions should be low.

- **Data Cleaning** ([Khanna et al., 2019](https://proceedings.mlr.press/v89/khanna19a.html)): Uses TDA to identify training samples responsible for misclassification. Removing them from the training set, we expect to see an improvement in the model performance when we retrain the model.

- **Mislabeled Data Detection** ([Koh and Liang, 2017](https://proceedings.mlr.press/v70/koh17a.html)): Computes the proportion of noisy training labels detected as a function of the percentage of inspected training samples. The samples are inspected in order according to their global TDA ranking, which is computed using local attributions. This produces a cumulative mislabeling detection curve. We expect to see a curve that rapidly increases as we check more of the training data, thus we compute the area under this curve

- **Mixed Datasets** ([Hammoudeh and Lowd, 2022](https://dl.acm.org/doi/abs/10.1145/3548606.3559335)): In a setting, where a model has been trained on two datasets: a clean dataset (e.g. CIFAR-10) and an adversarial (e.g. zeros from MNIST), this metric evaluates how well the model ranks the importance (attribution) of adversarial samples compared to clean samples when making predictions on an adversarial example. The evaluation is done using the Area Under the Precision-Recall Curve (AUPRC).

### Benchmarks

TODO: table with available benchmarks

## 🔬 Getting Started

### Installation

To install the latest release of **quanda** use:

```setup
pip install git+https://github.com/dilyabareeva/quanda.git
```

**quanda** requires Python 3.7 or later. It is recommended to use a virtual environment to install the package.

### Usage


In the following, we provide a quick guide to **quanda** usage. To begin using **quanda**, ensure you have the following:

- **Trained PyTorch Model (`model`)**: A PyTorch model that has already been trained on a relevant dataset.
- **PyTorch Dataset (`train_set`)**: The dataset used during the training of the model.
- **Test Batches (`test_tensor`) and Explanation Targets (`target`)**: A batch of test data (`test_tensor`) and the corresponding explanation targets (`target`). Generally, it is advisable to use the model's predicted labels as the targets. In the following, we use the `torch.utils.data.DataLoader` to load the test data in batches.


As an example, we will demonstrate the generation of explanations using `SimilarityInfluence` data attribution from `Captum` and the evaluation of these explanations using the **Model Randomization** metric.

<details>
<summary><b><big>Step 1. Import dependencies and library components</big></b></summary>

```python
import torch
from torch.utils.data import DataLoader
import tqdm

from quanda.explainers.wrappers import captum_similarity_explain, CaptumSimilarity
from quanda.metrics.randomization import ModelRandomizationMetric
```
</details>

<details>

<summary><b><big>Step 2. Define explanation parameters</big></b></summary>

While `explainer_cls` is passed directly to the metric, `explain` function is used to generate explanations fed to a metric.
```python
explainer_cls = CaptumSimilarity
explain = captum_similarity_explain
explain_fn_kwargs = {"layers": "avgpool"}
model_id = "default_model_id"
cache_dir = "./cache"
```
</details>

<details>

<summary><b><big>Step 3. Initialize metric</big></b></summary>

```python
model_rand = ModelRandomizationMetric(
        model=model,
        train_dataset=train_set,
        explainer_cls=explainer_cls,
        expl_kwargs=explain_fn_kwargs,
        model_id=model_id,
        cache_dir=cache_dir,
        correlation_fn="spearman",
        seed=42,
)
```
</details>

<details>
<summary><b><big>Step 4. Iterate over test set and feed tensor batches first to explain, then to metric</big></b></summary>

```python
for i, (data, target) in enumerate(tqdm(test_loader)):
    data, target = data.to(DEVICE), target.to(DEVICE)
    tda = explain(
        model=model,
        model_id=model_id,
        cache_dir=cache_dir,
        test_tensor=data,
        train_dataset=train_set,
        device=DEVICE,
        **explain_fn_kwargs,
    )
    model_rand.update(data, tda)

print("Model heuristics metric output:", model_rand.compute())
```
</details>

More detailed examples can be found in the following [tutorials](https://github.com/dilyabareeva/quanda/tree/main/tutorials) section.

### Custom Explainers

In addition to the already provided explainers, **quanda** allows users to integrate their own explainer methods. This section provides a guide on how to create a wrapper for a custom explainer.

<details>
<summary><b><big>Step 1. Create an explainer class</big></b></summary>

Your custom explainer should inherit from the base `Explainer` class provided by **quanda**. The first step is to initialize your custom explainer within the `__init__` method.
```python
from quanda.explainers.base import Explainer

class CustomExplainer(Explainer):
    def __init__(self, model, train_dataset, **kwargs):
        super().__init__(model, train_dataset, **kwargs)
        # Initialize your explainer here
```
</details>

<details>
<summary><b><big>Step 2. Implement the explain method</big></b></summary>

The core of your wrapper is the `explain` method. This function should take test samples and their corresponding target values as input and return a 2D tensor containing the influence scores.

- `test`: The test batch for which explanations are generated.
- `targets`: The target values for the explanations.

Ensure the output tensor has the shape `(test_samples, train_samples)`.

```python
def explain(self, test: torch.Tensor, targets: Union[List[int], torch.Tensor]) -> torch.Tensor:
    # Compute your influence scores here
    return influence_scores
 ```
</details>

<details>
<summary><b><big>Step 3. Implement the self_influence method (Optional) </big></b></summary>

By default, **quanda** includes a built-in method for calculating self-influence scores. However, you can override this method to provide a custom implementation. This method should calculate how much each training sample influences itself and return a tensor of the computed self-influence scores.

```python
def self_influence(self, batch_size: int = 1) -> torch.Tensor:
    # Compute your self-influence scores here
    return self_influence_scores
```
</details>

For detailed examples, we refer to the existing explainer wrappers in **quanda**:

- **Captum Explainers**: [captum_influence.py](https://github.com/dilyabareeva/quanda/tree/main/quanda/explainers/wrappers/captum_influence.py)
- **TRAK**: [trak_wrapper.py](https://github.com/dilyabareeva/quanda/tree/main/quanda/explainers/wrappers/trak_wrapper.py)
- **Representer Points**: [representer_points.py](https://github.com/dilyabareeva/quanda/tree/main/quanda/explainers/wrappers/representer_points.py)


## ⚠️ Usage Tips and Caveats

- **Controlled Setting Evaluation**: Many metrics require access to ground truth labels for datasets, such as the indices of the "shorcut samples" in the Shortcut Detection metric, or the mislabeling (noisy) label indices for the Mislabeling Detection Metric. However, users often may not have access to these labels. To address this, we recommend either using one of our pre-built benchmark (see Benchmarks section of this README) suites or generating (`generate` method) a custom benchmark for comparing explainers. Benchmarks provide a controlled environment for systematic evaluation.

- **Caching**: Many of the Explaienrs in our library generate re-usable cache. The `cache_id` and `model_id` parameters passed to various class instances are used to store these intermediary results. Please ensure that each experiment is assigned a unique combination of these arguments. Failing to do so could lead to incorrect reuse of cached results. If you wish to avoid re-using cached results, you can set the `load_from_disk` parameter to `False`.

- **Explainers Are Expensive To Calculate**: Certain Explainers, such as TracInCPRandomProj, may lead to OutOfMemory (OOM) issues when applied to large models or datasets. In such cases, we recommend adjusting memory usage by either reducing the dataset size or using smaller models to avoid these issues.
- 
## 📓 Tutorials

We have included a few  [tutorials](https://github.com/dilyabareeva/quanda/tree/main/tutorials) to demonstrate the usage of **quanda**:

* [Explainers](https://github.com/dilyabareeva/quanda/blob/main/tutorials/demo.ipynb): shows how different explainers can be produced with **quanda**
* [Metrics](https://github.com/dilyabareeva/quanda/blob/main/tutorials/demo_metrics.ipynb): demonstrates how to use the metrics in **quanda** to evaluate the performance of a model
* [Benchmarks](https://github.com/dilyabareeva/quanda/blob/main/tutorials/demo_benchmarks.ipynb): shows how to use the benchmarking tools in **quanda** to evaluate a data attribution method

To install the library with tutorials dependencies, run:

```bash
pip install -e '.[tutorials]'
```

## 👩‍💻Contributing
We welcome contributions to **quanda**! You could contribute by:
- Opening an issue to report a bug or request a feature
- Submitting a pull request to fix a bug, add a new explainer wrapper, a new metric, or other feature.

A detailed guide on how to contribute to **quanda** can be found [here](https://github.com/dilyabareeva/quanda/blob/main//CONTRIBUTING.md).

To set up the development environment, clone the repository and install the dependencies:

```bash
pip install -e '.[dev]'
pip uninstall quanda
```

Install the pre-commit hooks:
```bash
pre-commit install
```

Alternatively, run the makefile before a commit to ensure the code is formatted and linted correctly:
```bash
make style
```

To run the tests:
```bash
pytest
```


If you have any questions, please [open an issue](https://github.com/dilyabareeva/quanda/issues/new)
or write us at [dilyabareeva@gmail.com](mailto:dilyabareeva@gmail.com) or [galip.uemit.yolcu@hhi.fraunhofer.de](mailto:galip.uemit.yolcu@hhi.fraunhofer.de).
