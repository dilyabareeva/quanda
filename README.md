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


![py_versions](https://github.com/dilyabareeva/quanda/assets/44092813/36499a1d-aefb-455f-b73a-57ca4794f31f)<p align="center">

## Table of contents

* [Installation](#installation)
* [Usage](#usage)

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
Clone the repository and run the following inside the repo root to install the dependencies:

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
