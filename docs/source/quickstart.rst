Quickstart
===============

Installation
------------

To install the latest release of |quanda|, use the following command in your terminal:

.. code-block:: console

   (.venv) $ pip install quanda
   (.venv) $ pip install captum@git+https://github.com/pytorch/captum

|quanda| requires Python 3.7 or later. It is recommended to use a virtual environment to install the package.

.. note::
   In the examples that follow, we will demonstrate the generation of explanations generated using ``SimilarityInfluence`` data attributor from ``Captum``.

Using Metrics
-------------
To begin using |quanda| metrics, you need the following components:

- **Trained PyTorch Model (** ``model`` **)**: A PyTorch model that has already been trained on a relevant dataset. As a placeholder, we used the layer name "avgpool" below. Please replace it with the name of one of the layers in your model.
- **PyTorch Dataset (** ``train_set`` **)**: The dataset used during the training of the model.
- **Test Dataset (** ``eval_set`` **)**: The dataset to be used as test inputs for generating explanations. Explanations are generated with respect to an output neuron corresponding to a certain class. This class can be selected to be the ground truth label of the test points, or the classes predicted by the model. In the following we will use the predicted labels to generate explanations.
Next, we demonstrate how to evaluate explanations using the **Model Randomization** metric.

**1. Import dependencies and library components**

.. code-block:: python

   from torch.utils.data import DataLoader
   from tqdm import tqdm

   from quanda.explainers.wrappers import CaptumSimilarity
   from quanda.metrics.heuristics import ModelRandomizationMetric

**2. Create the explainer object**

We now create our explainer. The device to be used by the explainer and metrics is inherited from the model, thus we set the model device explicitly.

.. code-block:: python

   DEVICE = "cpu"
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

The ``ModelRandomizationMetric`` needs to instantiate a new explainer to generate explanations for a randomized model. These will be compared with the explanations of the original model. Therefore, ``explainer_cls`` is passed directly to the metric along with initialization parameters of the explainer for the randomized model.

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

**4. Iterate over test set to generate explanations and update the metric**

We now start producing explanations with our TDA method. We go through the test set batch-by-batch. For each batch, we first generate the attributions using the predicted labels, and we then update the metric with the produced explanations to showcase how to concurrently handle the explanation and evaluation processes.

.. code-block:: python

   test_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)
   for test_tensor, _ in tqdm(test_loader):
      test_tensor = test_tensor.to(DEVICE)
      target = model(test_tensor).argmax(dim=-1)
      tda = explainer.explain(
         test_tensor=test_tensor,
         targets=target
      )
      model_rand.update(test_data=test_tensor, explanations=tda, explanation_targets=target)

   print("Randomization metric output:", model_rand.compute())

Using Benchmarks
----------------

Using Pre-assembled Benchmarks
++++++++++++++++++++++++++++++
The pre-assembled benchmarks allow us to streamline the evaluation process by downloading the necessary data and models, and running the evaluation in a single command. The following code demonstrates how to use the ``mnist_subclass_detection`` benchmark:

**1. Import dependencies and library components**

.. code-block:: python

   from quanda.explainers.wrappers import CaptumSimilarity
   from quanda.benchmarks.downstream_eval import SubclassDetection

**2. Prepare arguments for the explainer object**

.. code-block:: python

   DEVICE = "cpu"
   model.to(DEVICE)

   explainer_kwargs = {
      "layers": "avgpool",
      "model_id": "default_model_id",
      "cache_dir": "./cache"
   }

**3. Load a pre-assembled benchmark and score an explainer**

.. code:: python

   subclass_detect = SubclassDetection.download(
      name="mnist_subclass_detection",
      cache_dir=cache_dir,
      device="cpu",
   )
   score = subclass_detect.evaluate(
      explainer_cls=CaptumSimilarity,
      expl_kwargs=explain_fn_kwargs,
      batch_size=batch_size,
   )["score"]
   print(f"Subclass Detection Score: {score}")

Assembling a benchmark from existing components
+++++++++++++++++++++++++++++++++++++++++++++++

Next, we demonstrate assembling a benchmark with assets that the user has prepared. As in the `Using Metrics`_ section, we will assume that the user has already trained ``model`` on ``train_set``, and a corresponding ``eval_set`` to be used for generating and evaluating explanations.

**1. Import dependencies and library components**

 .. code-block:: python

   from quanda.explainers.wrappers import CaptumSimilarity
   from quanda.benchmarks.ground_truth import TopKCardinality

**2. Prepare arguments for the explainer object**

.. code-block:: python

   DEVICE = "cpu"
   model.to(DEVICE)

   explainer_kwargs = {
      "layers": "avgpool",
      "model_id": "default_model_id",
      "cache_dir": "./cache"
   }

**3. Assemble the benchmark object and run the evaluation**

We now have everything we need, we can just assemble the benchmark and run it. This will encapsulate the process of instantiating the explainer, generating explanations and using the :doc:`TopKCardinalityMetric <docs_api/quanda.metrics.heuristics.topk_cardinality.TopKCardinalityMetric>` to evaluate them.

.. code-block:: python

   topk_cardinality = TopKCardinality.assemble(
      model=model,
      train_dataset=train_set,
      eval_dataset=eval_set,
   )
   score = topk_cardinality.evaluate(
      explainer_cls=CaptumSimilarity,
      expl_kwargs=explain_fn_kwargs,
      batch_size=batch_size,
   )["score"]
   print(f"Top K Cardinality Score: {score}")


Generating the benchmark object from scratch
++++++++++++++++++++++++++++++++++++++++++++

Some evaluation strategies require a controlled setup or a different strategy of using attributors to evaluate them. For example, the :doc:`MislabelingDetectionMetric <docs_api/quanda.metric.downstream_eval.mislabeling_detection>` requires a dataset with known mislabeled examples. It computes the self-influence of training points to evaluate TDA methods. Therefore, it is fairly complicated to train a model on a mislabeled dataset, and then using the metric object or assembling a benchmark object to run the evaluation. While pre-assembled benchmarks allow to use pre-computed assets, |quanda| :doc:`Benchmark <docs_api/quanda.benchmarks.base>` objects provide the `generate` interface, which allows the user to prepare this setup from scratch.

As in previous examples, we assume that ``train_set`` refers to  a vanilla training dataset, without any modifications for evaluation. Furthermore, we assume ``model`` refers to a torch ``Module``, but in this example we do not require that ``model`` is trained. Finally, ``n_classes`` is the number of classes in the ``train_set``.

**1. Import dependencies and library components**

.. code-block:: python

   import torch

   from quanda.explainers.wrappers import CaptumSimilarity
   from quanda.benchmarks.downstream_eval import MislabelingDetection

**2. Prepare arguments for the explainer object**

.. code-block:: python

   DEVICE = "cpu"
   model.to(DEVICE)

   explainer_kwargs = {
      "layers": "avgpool",
      "model_id": "default_model_id",
      "cache_dir": "./cache"
   }

**3. Prepare the trainer**

For mislabeling detection, we will train a model from scratch. |quanda| allows to use Lightning ``Trainer`` objects. If you want to use Lightning trainers, ``model`` needs to be an instance of a Lightning ``LightningModule``. Alternatively, you can use an instance of :doc:`quanda.utils.training.BaseTrainer <docs_api/quanda.utils.training.trainer>`. In this example, we use a very simple training setup via the :doc:`quanda.utils.training.Trainer <quanda.utils.training.trainer>` class.

.. code-block:: python
   trainer = Trainer(
      max_epochs=100,
      optimizer=torch.optim.SGD,
      lr=0.01,
      criterion=torch.nn.CrossEntropyLoss(),
   )

4. Generate the benchmark object and run the evaluation</big></b></summary>

We can now call the ``generate`` method to instantiate our :doc:`MislabelingDetection <docs_api/quanda.benchmarks.downstream_eval.mislabeling_detection>` object and directly start the evaluation process with it. The ``generate`` method takes care of model training using ``trainer``, generation of explanations and their evaluation.

.. code-block:: python

   mislabeling_detection = MislabelingDetection.generate(
      model=model,
      base_dataset=train_set,
      n_classes=n_classes,
      trainer=trainer,
   )
   score = mislabeling_detection.evaluate(
      explainer_cls=CaptumSimilarity,
      expl_kwargs=explain_fn_kwargs,
      batch_size=batch_size,
   )["score"]
   print(f"Mislabeling Detection Score: {score}")

More detailed examples can be found in the :doc:`tutorials <./tutorials>` page.

Custom Explainers
-----------------

In addition to the built-in explainers, |quanda| supports the evaluation of custom explainer methods. This section provides a guide on how to create a wrapper for a custom explainer that matches our interface.

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

For detailed examples, we refer to the :doc:`existing explainer wrappers <docs_api/quanda.explainers.wrappers>` in |quanda|.

⚠️ Usage Tips and Caveats
+++++++++++++++++++++++++

-  **Controlled Setting Evaluation**: Many metrics require access to ground truth labels for datasets, such as the indices of the “shorcut samples” in the Shortcut Detection metric, or the mislabeling (noisy) label indices for the Mislabeling Detection Metric. However, users often may not have access to these labels. To address this, we recommend either using one of our pre-built benchmark suites or generating (using the ``generate`` method) a custom benchmark for comparing explainers. Benchmarks provide a controlled environment for systematic evaluation.

-  **Caching**: Many explainers in our library generate re-usable cache. The ``cache_id`` and ``model_id`` parameters passed to various class instances are used to store these intermediary results. Ensure each experiment is assigned a unique combination of these arguments. Failing to do so could lead to incorrect reuse of cached results. If you wish to avoid re-using cached results, you can set the ``load_from_disk`` parameter to ``False``.

-  **Explanations Are Expensive To Compute**: Certain explainers, such as TracInCPRandomProj, may lead to OutOfMemory (OOM) issues when applied to large models or datasets. In such cases, we recommend adjusting memory usage by either reducing the dataset size or using smaller models to avoid these issues.
