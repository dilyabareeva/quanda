<p align="center">
 <img width="350" alt="QuanDA" src="https://github.com/dilyabareeva/data_attribution_evaluation/assets/44092813/e5ffbeea-aeb9-4b82-939a-e5efa1179140">
</p>
<!--<h1 align="center"><b>QuanDA</b></h1>-->
<p align="center">A toolkit for <b>quan</b>titative evaluation of <b>d</b>ata <b>a</b>ttribution methods.</p>
<p align="center">
  PyTorch


## Table of contents

* [Installation](#installation)
* [Usage](#usage)

## Installation


To install
<span style="color: #4D4352; font-family: 'arial narrow', arial, sans-serif;">
quanda
</span>:

```setup
pip install git+https://github.com/dilyabareeva/quanda.git
```

## Usage

An excerpt from `tutorials/usage_testing.py`:
```python

from src.explainers.wrappers.captum_influence import captum_similarity_explain
from src.metrics.localization.identical_class import IdenticalClass
from src.metrics.randomization.model_randomization import (
    ModelRandomizationMetric,
)
from src.metrics.unnamed.top_k_overlap import TopKOverlap

# define explanation parameters
explain = captum_similarity_explain
explain_fn_kwargs = {"layers": "avgpool"}
model_id = "default_model_id"
cache_dir = "./cache"

# initialize metrics
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

# iterate over test set and feed tensor batches first to explain, then to metric
for i, (data, target) in enumerate(tqdm(test_loader)):
    data, target = data.to(DEVICE), target.to(DEVICE)

    # some metrics have an explain_update() method in addition to update():
    model_rand.update(data)

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
