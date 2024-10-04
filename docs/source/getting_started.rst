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

- **Trained PyTorch Model (`model`)**: A PyTorch model that has already been trained on a relevant dataset. As a placeholder, we used the layer name "avgpool" below. Please replace it with the name of one of the layers in your model.
- **PyTorch Dataset (`train_set`)**: The dataset used during the training of the model.
- **Test Batches (`test_tensor`) and Explanation Targets (`target`)**: A batch of test data (`test_tensor`) and the corresponding explanation targets (`target`). Generally, it is advisable to use the model's predicted labels as the targets. In the following, we use the `torch.utils.data.DataLoader` to load the test data in batches.

As an example, we will demonstrate the generation of explanations generated using `SimilarityInfluence` data attribution from `Captum` and the evaluation of these explanations using the **Model Randomization** metric.

**1. Import dependencies and library components**

.. code-block:: python

   import torch
   from torch.utils.data import DataLoader
   from tqdm import tqdm

   from quanda.explainers.wrappers import captum_similarity_explain, CaptumSimilarity
   from quanda.metrics.heuristics import ModelRandomizationMetric

**2. Create the explainer object**

We now create our explainer. The device to be used by the explainer and metrics is inherited from the model, thus we set the model device explicitly.

.. code-block:: python

   DEVICE="cpu"
   model.to(DEVICE)
   explainer_kwargs = {
      "layers": "avgpool",
      "model_id": "default_model_id",
      "cache_dir": "./cache"
   }
   explainer = CaptumSimilarity(
      model=model,
      train_dataset=train_set,
      **explainer_kwargs
   )

**3. Initialize the metric**


The `ModelRandomizationMetric` needs to instantiate a new explainer to generate explanations for a randomized model. These will be compared with the explanations of the original model. Therefore, `explainer_cls` is passed directly to the metric along with initialization parameters of the explainer.

.. code-block:: python

   explainer_kwargs = {
      "layers": "avgpool",
      "model_id": "randomized_model_id",
      "cache_dir": "./cache"
   }
   model_rand = ModelRandomizationMetric(
         model=model,
         train_dataset=train_set,
         explainer_cls=CaptumSimilarity,
         expl_kwargs=explainer_kwargs,
         correlation_fn="spearman",
         seed=42,
   )

**4. Iterate over test set and feed tensor batches first to explain, then to metric**

.. code-block:: python

   for i, (test_tensor, target) in enumerate(tqdm(test_loader)):
      test_tensor, target = test_tensor.to(DEVICE), target.to(DEVICE)
      tda = explainer.explain(
         test=test_tensor,
         targets=target
      )
      model_rand.update(test_data=test_tensor, explanations=tda, explanation_targets=target)

   print("Model heuristics metric output:", model_rand.compute())
