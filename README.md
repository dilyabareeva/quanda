<p align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/e4dd026b-1325-4368-9fd5-9f2e67767371">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/d07deeba-8a76-40f6-910d-0523897b2a29">
  <img width="700" alt="quanda" src="https://github.com/user-attachments/assets/d07deeba-8a76-40f6-910d-0523897b2a29g">
</picture>
</p>



  <source width="700" media="(prefers-color-scheme: light)" srcset="https://github.com/user-attachments/assets/e4dd026b-1325-4368-9fd5-9f2e67767371">
  <source width="700" media="(prefers-color-scheme: dark)" srcset="https://github.com/user-attachments/assets/d07deeba-8a76-40f6-910d-0523897b2a29">
  
<p align="center">Toolkit for <b>quan</b>titative evaluation of <b>d</b>ata <b>a</b>ttribution methods.</p>
<p align="center">
  PyTorch
</p>


![py_versions](https://github.com/dilyabareeva/quanda/assets/44092813/36499a1d-aefb-455f-b73a-57ca4794f31f)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

_quanda is currently under active development so carefully note the quanda release version to ensure reproducibility of your work._


## Table of contents


* [Library overview](#library-overview)
* [Installation](#installation)
* [Tutorials](#tutorials)
* [Contributing](#contributing)

## Library overview
**Training data attribution** (TDA) methods attribute model output to its training samples ([Koh and Liang, 2017](https://proceedings.mlr.press/v70/koh17a.html); [Yeh et al., 2018](https://proceedings.neurips.cc/paper_files/paper/2018/file/8a7129b8f3edd95b7d969dfc2c8e9d9d-Paper.pdf); [Park et al., 2023](https://proceedings.mlr.press/v202/park23c.html); [Pruthi et al., 2020](https://proceedings.neurips.cc/paper_files/paper/2020/file/e6385d39ec9394f2f3a354d9d2b88eec-Paper.pdf); [Bae et al., 2024](https://arxiv.org/abs/2405.12186)). Outside of being used for understanding models, TDA has also found usage in a large variety of applications such as debugging model behavior ([Koh and Liang, 2017](https://proceedings.mlr.press/v70/koh17a.html); [Yeh et al., 2018](https://proceedings.neurips.cc/paper_files/paper/2018/file/8a7129b8f3edd95b7d969dfc2c8e9d9d-Paper.pdf); [K and Søgaard, 2021](https://arxiv.org/abs/2111.04683); [Guo et al., 2021](https://aclanthology.org/2021.emnlp-main.808)), data summarization ([Khanna et al., 2019](https://proceedings.mlr.press/v89/khanna19a.html); [Marion et al., 2023](https://openreview.net/forum?id=XUIYn3jo5T); [Yang et al., 2023](https://openreview.net/forum?id=4wZiAXD29TQ)), dataset selection ([Engstrom et al., 2024](https://openreview.net/forum?id=GC8HkKeH8s); [Chhabra et al., 2024](https://openreview.net/forum?id=HE9eUQlAvo)), fact tracing ([Akyurek et al., 2022](https://aclanthology.org/2022.findings-emnlp.180)) and machine unlearning ([Warnecke
et al., 2023](https://arxiv.org/abs/2108.11577)).

### Metrics

- **Identical Class / Identical Subclass** ([Hanawa et al., 2021](https://openreview.net/forum?id=9uvhpyQwzM_)): Measures the proportion of identical classes or subclasses in the top-1 training samples.

-  **Top-K Overlap**  ([Hanawa et al., 2021](https://openreview.net/forum?id=9uvhpyQwzM_)): Measures the cardinality of the union of the top-K training samples.

- **Model Randomization** ([Hanawa et al., 2021](https://openreview.net/forum?id=9uvhpyQwzM_)): Measures the correlation between the original TDA and the TDA of a model with randomized weights.

- **Data Cleaning** ([Khanna et al., 2019](https://proceedings.mlr.press/v89/khanna19a.html)): Uses TDA to identify training samples responsible for misclassification, removing them from the training set, retraining the model, and measuring the change in model performance.

- **Mislabeled Data Detection** ([Koh and Liang, 2017](https://proceedings.mlr.press/v70/koh17a.html)): Measures the proportion of noisy training labels detected as a function of the percentage of inspected training samples, where the samples are inspected in order according to their global TDA ranking.



## Installation


To install
<span style="color: #4D4352; font-family: 'arial narrow', arial, sans-serif;">
quanda</span>:

```setup
pip install git+https://github.com/dilyabareeva/quanda.git
```

## Usage

Excerpts from `tutorials/usage_testing.py`:

<details>
<summary><b><big>Step 1. Import library components</big></b></summary>

```python
from quanda.explainers.wrappers import captum_similarity_explain, CaptumSimilarity
from quanda.metrics.localization import ClassDetectionMetric
from quanda.metrics.randomization import ModelRandomizationMetric
from quanda.metrics.unnamed.top_k_overlap import TopKOverlapMetric
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

<summary><b><big>Step 3. Initialize metrics</big></b></summary>

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
        device=DEVICE,
)

id_class = IdenticalClass(model=model, train_dataset=train_set, device=DEVICE)

top_k = TopKOverlap(model=model, train_dataset=train_set, top_k=1, device=DEVICE)

# dataset cleaning
pl_module = BasicLightningModule(
    model=copy.deepcopy(model),
    optimizer=torch.optim.SGD,
    lr=0.01,
    criterion=torch.nn.CrossEntropyLoss(),
)
trainer = Trainer.from_lightning_module(model, pl_module)

data_clean = DatasetCleaning(
    model=model,
    train_dataset=train_set,
    global_method="sum_abs",
    trainer=trainer,
    trainer_fit_kwargs={"max_epochs": 3},
    top_k=50,
    device=DEVICE,
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
    id_class.update(target, tda)
    top_k.update(tda)
    data_clean.update(tda)

print("Model randomization metric output:", model_rand.compute())
print("Identical class metric output:", id_class.compute())
print("Top-k overlap metric output:", top_k.compute())

print("Dataset cleaning metric computation started...")
print("Dataset cleaning metric output:", data_clean.compute())
```
</details>

## Contribution
We welcome contributions to quanda! You could contribute by:
- Opening an issue to report a bug or request a feature
- Submitting a pull request to fix a bug, add a new explainer wrapper, a new metric, or other feature.

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