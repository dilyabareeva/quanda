<p align="center">
 <img width="350" alt="QuanDA" src="https://github.com/dilyabareeva/data_attribution_evaluation/assets/44092813/e5ffbeea-aeb9-4b82-939a-e5efa1179140">
</p>
<!--<h1 align="center"><b>QuanDA</b></h1>-->
<p align="center">A toolkit for <b>quan</b>titative evaluation of <b>d</b>ata <b>a</b>ttribution methods.</p>
<p align="center">
  PyTorch

<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="191.6" height="20"><linearGradient id="smooth" x2="0" y2="100%"><stop offset="0" stop-color="#bbb" stop-opacity=".1"/><stop offset="1" stop-opacity=".1"/></linearGradient><clipPath id="round"><rect width="191.6" height="20" rx="3" fill="#fff"/></clipPath><g clip-path="url(#round)"><rect width="65.5" height="20" fill="#555"/><rect x="65.5" width="126.1" height="20" fill="#007ec6"/><rect width="191.6" height="20" fill="url(#smooth)"/></g><g fill="#fff" text-anchor="middle" font-family="DejaVu Sans,Verdana,Geneva,sans-serif" font-size="110"><image x="5" y="3" width="14" height="14" xlink:href="https://dev.w3.org/SVG/tools/svgweb/samples/svg-files/python.svg"/><text x="422.5" y="150" fill="#010101" fill-opacity=".3" transform="scale(0.1)" textLength="385.0" lengthAdjust="spacing">python</text><text x="422.5" y="140" transform="scale(0.1)" textLength="385.0" lengthAdjust="spacing">python</text><text x="1275.5" y="150" fill="#010101" fill-opacity=".3" transform="scale(0.1)" textLength="1161.0" lengthAdjust="spacing">3.8 | 3.9 | 3.10 | 3.11</text><text x="1275.5" y="140" transform="scale(0.1)" textLength="1161.0" lengthAdjust="spacing">3.8 | 3.9 | 3.10 | 3.11</text><a xlink:href="https://www.python.org/"><rect width="65.5" height="20" fill="rgba(0,0,0,0)"/></a><a xlink:href="https://www.python.org/"><rect x="65.5" width="126.1" height="20" fill="rgba(0,0,0,0)"/></a></g></svg>

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
