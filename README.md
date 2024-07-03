<p align="center">
 <img width="350" alt="QuanDA" src="https://github.com/dilyabareeva/data_attribution_evaluation/assets/44092813/e5ffbeea-aeb9-4b82-939a-e5efa1179140">
</p>
<!--<h1 align="center"><b>QuanDA</b></h1>-->
<p align="center">A toolkit for <b>quan</b>titative evaluation of <b>d</b>ata <b>a</b>ttribution methods.</p>
<p align="center">
  PyTorch

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
from src.explainers.wrappers.captum_influence import captum_similarity_explain
from src.metrics.localization.identical_class import IdenticalClass
from src.metrics.randomization.model_randomization import (
    ModelRandomizationMetric,
)
from src.metrics.unnamed.top_k_overlap import TopKOverlap
```
</details>

<details>

<summary><b><big>Step 2. Define explanation parameters</big></b></summary>

```python
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
    explain_fn=explain,
    explain_fn_kwargs=explain_fn_kwargs,
    model_id=model_id,
    cache_dir=cache_dir,
    correlation_fn="spearman",
    seed=42,
    device=DEVICE,
)

id_class = IdenticalClass(model=model, train_dataset=train_set, device=DEVICE)

top_k = TopKOverlap(model=model, train_dataset=train_set, top_k=1, device="cpu")
```
</details>

<details>
<summary><b><big>Step 4. Iterate over test set and feed tensor batches first to explain, then to metric</big></b></summary>

```python
for i, (data, target) in enumerate(tqdm(test_loader)):
    data, target = data.to(DEVICE), target.to(DEVICE)

    # some metrics have an explain_update() method in addition to update():
    model_rand.explain_update(data)

    # metrics that do not generate explanations only have an update() method:
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
    top_k.update(target)
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
