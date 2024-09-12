Getting Started
===============

Installation
------------

To install the latest release of **quanda**, use the following command in your terminal:

.. code-block:: console

   (.venv) $ pip install git+https://github.com/dilyabareeva/quanda.git

**quanda** requires Python 3.7 or later. It is recommended to use a virtual environment to install the package.

Basic Usage
-----------

In the following, we provide a quick guide to **quanda** usage. To begin using **quanda**, ensure you have the following:

- **Trained PyTorch Model (`model`)**: A PyTorch model that has already been trained on a relevant dataset.
- **PyTorch Dataset (`train_set`)**: The dataset used during the training of the model.
- **Test Batches (`test_tensor`) and Explanation Targets (`target`)**: A batch of test data (`test_tensor`) and the corresponding explanation targets (`target`). Generally, it is advisable to use the model's predicted labels as the targets. In the following, we use the `torch.utils.data.DataLoader` to load the test data in batches.

As an example, we will demonstrate the generation of explanations generated using `SimilarityInfluence` data attribution from `Captum` and the evaluation of these explanations using the **Model Randomization** metric.

**1. Import dependencies and library components**

.. code-block:: python

   import torch
   from torch.utils.data import DataLoader
   import tqdm

   from quanda.explainers.wrappers import captum_similarity_explain, CaptumSimilarity
   from quanda.metrics.randomization import ModelRandomizationMetric

**2. Define explanation parameters**

While `explainer_cls` is passed directly to the metric, `explain` function is used to generate explanations fed to a metric.

.. code-block:: python

   explainer_cls = CaptumSimilarity
   explain = captum_similarity_explain
   explain_fn_kwargs = {"layers": "avgpool"}
   model_id = "default_model_id"
   cache_dir = "./cache"

**3. Initialize metric**

.. code-block:: python

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

**4. Iterate over test set and feed tensor batches first to explain, then to metric**

.. code-block:: python

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
