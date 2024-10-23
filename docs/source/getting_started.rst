Getting Started
===============

Installation
------------

To install the latest release of |quanda|, use the following command in your terminal:

.. code-block:: console

   (.venv) $ pip install quanda
   (.venv) $ pip install captum@git+https://github.com/pytorch/captum

|quanda| requires Python 3.7 or later. It is recommended to use a virtual environment to install the package.


Basic Usage
-----------

In the following, we provide a quick guide to |quanda| usage. To begin using |quanda|, ensure you have the following:

- **Trained PyTorch Model (`model`)**: A PyTorch model that has already been trained on a relevant dataset. As a placeholder, we used the layer name "avgpool" below. Please replace it with the name of one of the layers in your model.
- **PyTorch Dataset (`train_set`)**: The dataset used during the training of the model.
- **Test Batches (`test_tensor`) and Explanation Targets (`target`)**: A batch of test data (`test_tensor`) and the corresponding explanation targets (`target`). Generally, it is advisable to use the model's predicted labels as the targets. In the following, we use the `torch.utils.data.DataLoader` to load the test data in batches.

.. note::
   In the examples that follow, we will demonstrate the generation of explanations generated using ``SimilarityInfluence`` data attribution from ``Captum`` and the evaluation of these explanations using the **Model Randomization** metric.

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
         test_tensor=test_tensor,
         targets=target
      )
      model_rand.update(test_data=test_tensor, explanations=tda, explanation_targets=target)

   print("Model heuristics metric output:", model_rand.compute())

Using Benchmarks
++++++++++++++++
The pre-assembled benchmarks allow us to streamline the evaluation process by downloading the necessary data and models, and running the evaluation in a single command. **Step 1** and **Step 2** from the previous section are still required to be executed before running the benchmark. The following code demonstrates how to use the ``mnist_subclass_detection`` benchmark:

**Step 3. Load a pre-assembled benchmark and score an explainer**

.. code:: python

   subclass_detect = SubclassDetection.download(
       name=`mnist_subclass_detection`,
       cache_dir=cache_dir,
       device="cpu",
   )
   score = dst_eval.evaluate(
       explainer_cls=CaptumSimilarity,
       expl_kwargs=explain_fn_kwargs,
       batch_size=batch_size,
   )["score"]
   print(f"Subclass Detection Score: {score}")

More detailed examples can be found in the :doc:`tutorials <./tutorials>` page.

Custom Explainers
+++++++++++++++++

In addition to the built-in explainers, |quanda| supports the evaluatioon of custom explainer methods. This section provides a guide on how to create a wrapper for a custom explainer that matches our interface.

**Step 1. Create an explainer class**

Your custom explainer should inherit from the base :doc:`Explainer <docs_api/quanda.explainers.base>` class provided by |quanda|. The first step is to initialize your custom explainer within the ``__init__`` method.

.. code:: python

   from quanda.explainers.base import Explainer

   class CustomExplainer(Explainer):
       def __init__(self, model, train_dataset, **kwargs):
           super().__init__(model, train_dataset, **kwargs)
           # Initialize your explainer here

**Step 2. Implement the explain method**

The core of your wrapper is the ``explain`` method. This function should
take test samples and their corresponding target values as input and
return a 2D tensor containing the influence scores.

-  ``test``: The test batch for which explanations are generated.
-  ``targets``: The target values for the explanations.

You must ensure that the output tensor has the shape ``(test_samples, train_samples)``, where the entries in the train samples dimension are ordered in the same order as in the ``train_dataset`` that is being attributed.

.. code:: python

   def explain(
     self,
     test_tensor: torch.Tensor,
     targets: Union[List[int], torch.Tensor]
   ) -> torch.Tensor:
       # Compute your influence scores here
       return influence_scores


**Step 3. Implement the self_influence method (Optional)**

By default, |quanda| includes a built-in method for calculating self-influence scores. This base implementation computes all attributions over the training dataset, and collects the diagonal values in the attribution matrix. However, you can override this method to provide a more efficient implementation. This method should calculate how much each training sample influences itself and return a tensor of the computed self-influence scores.

.. code:: python

   def self_influence(self, batch_size: int = 1) -> torch.Tensor:
       # Compute your self-influence scores here
       return self_influence_scores

For detailed examples, we refer to the :doc:`existing explainer wrappers <./explainers>` in |quanda|.

⚠️ Usage Tips and Caveats
+++++++++++++++++++++++++

-  **Controlled Setting Evaluation**: Many metrics require access to ground truth labels for datasets, such as the indices of the “shorcut samples” in the Shortcut Detection metric, or the mislabeling (noisy) label indices for the Mislabeling Detection Metric. However, users often may not have access to these labels. To address this, we recommend either using one of our pre-built benchmark suites or generating (using the ``generate`` method) a custom benchmark for comparing explainers. Benchmarks provide a controlled environment for systematic evaluation.

-  **Caching**: Many explainers in our library generate re-usable cache. The ``cache_id`` and ``model_id`` parameters passed to various class instances are used to store these intermediary results. Ensure each experiment is assigned a unique combination of these arguments. Failing to do so could lead to incorrect reuse of cached results. If you wish to avoid re-using cached results, you can set the ``load_from_disk`` parameter to ``False``.

-  **Explanations Are Expensive To Compute**: Certain explainers, such as TracInCPRandomProj, may lead to OutOfMemory (OOM) issues when applied to large models or datasets. In such cases, we recommend adjusting memory usage by either reducing the dataset size or using smaller models to avoid these issues.
