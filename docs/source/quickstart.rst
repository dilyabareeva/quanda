Quickstart
===============

Installation
------------

To install the latest release of |quanda|, use the following command in your terminal:

.. code-block:: console

   (.venv) $ pip install quanda

|quanda| requires Python 3.10 or 3.11. It is recommended to use a virtual environment to install the package.

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

.. literalinclude:: ../../tests/integration/test_quickstart.py
   :language: python
   :start-after: # START2
   :end-before: # END2
   :dedent:

**3. Initialize the metric**

The ``ModelRandomizationMetric`` needs to instantiate a new explainer to generate explanations for a randomized model. These will be compared with the explanations of the original model. Therefore, ``explainer_cls`` is passed directly to the metric along with initialization parameters of the explainer for the randomized model.

.. literalinclude:: ../../tests/integration/test_quickstart.py
   :language: python
   :start-after: # START3
   :end-before: # END3
   :dedent:

**4. Iterate over test set to generate explanations and update the metric**

We now start producing explanations with our TDA method. We go through the test set batch-by-batch. For each batch, we first generate the attributions using the predicted labels, and we then update the metric with the produced explanations to showcase how to concurrently handle the explanation and evaluation processes.

.. literalinclude:: ../../tests/integration/test_quickstart.py
   :language: python
   :start-after: # START4
   :end-before: # END4
   :dedent:

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

.. literalinclude:: ../../tests/integration/test_quickstart.py
   :language: python
   :start-after: # START6
   :end-before: # END6
   :dedent:

**3. Load a pre-assembled benchmark and score an explainer**

.. literalinclude:: ../../tests/integration/test_quickstart.py
   :language: python
   :start-after: # START7_1
   :end-before: # END7_1
   :dedent:

.. literalinclude:: ../../tests/integration/test_quickstart.py
   :language: python
   :start-after: # START7_2
   :end-before: # END7_2
   :dedent:

Loading a benchmark from a configuration file
+++++++++++++++++++++++++++++++++++++++++++++++

Next, we demonstrate loading a benchmark from a YAML configuration file. As in the `Using Metrics`_ section, we will assume that the user has already trained ``model`` on ``train_set``, and a corresponding ``eval_set`` to be used for generating and evaluating explanations.

**1. Import dependencies and library components**

.. code-block:: python

   import yaml

   from quanda.explainers.wrappers import CaptumSimilarity
   from quanda.benchmarks.heuristics import TopKCardinality

**2. Prepare arguments for the explainer object**

.. literalinclude:: ../../tests/integration/test_quickstart.py
   :language: python
   :start-after: # START9
   :end-before: # END9
   :dedent:

**3. Load the benchmark from config and run the evaluation**

We now have everything we need: we can load the benchmark from a YAML configuration file and run the evaluation. This will encapsulate the process of instantiating the explainer, generating explanations and using the :doc:`TopKCardinalityMetric <docs_api/quanda.metrics.heuristics.top_k_cardinality>` to evaluate them.

.. literalinclude:: ../../tests/integration/test_quickstart.py
   :language: python
   :start-after: # START10
   :end-before: # END10
   :dedent:

Training the benchmark from scratch
+++++++++++++++++++++++++++++++++++

While we provide a number of benchmarks with pre-computed assets, |quanda| :doc:`Benchmark <docs_api/quanda.benchmarks.base>` objects also expose a ``train`` interface for preparing benchmarks from scratch. To train a benchmark, specify its components in a single YAML file (see ``quanda/benchmarks/resources/configs``).

For example, the :doc:`MislabelingDetection <docs_api/quanda.benchmarks.downstream_eval.mislabeling_detection>` benchmark requires a dataset with known mislabeled examples. The ``train`` method takes care of model training on the mislabeled dataset, and prepares the benchmark for evaluation.

**1. Import dependencies and library components**

.. code-block:: python

   import yaml

   from quanda.explainers.wrappers import CaptumSimilarity
   from quanda.benchmarks.downstream_eval import MislabelingDetection

**2. Prepare arguments for the explainer object**

.. literalinclude:: ../../tests/integration/test_quickstart.py
   :language: python
   :start-after: # START12
   :end-before: # END12
   :dedent:

**3. Train the benchmark**

For mislabeling detection, we will train a model from scratch using a dataset with a portion of labels flipped.

.. literalinclude:: ../../tests/integration/test_quickstart.py
   :language: python
   :start-after: # START13_2
   :end-before: # END13_2
   :dedent:

**4. Run the evaluation**

We can now call the ``evaluate`` method to directly start the evaluation process on the benchmark.

.. literalinclude:: ../../tests/integration/test_quickstart.py
   :language: python
   :start-after: # START14
   :end-before: # END14
   :dedent:

More detailed examples can be found in the :doc:`tutorials <./tutorials>` page.
You can also use Hydra for benchmark training configuration, as shown in `scripts/train.py`.

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
     test_data: torch.Tensor,
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

Usage Tips and Caveats
++++++++++++++++++++++

-  **Controlled Setting Evaluation**: Many metrics require access to ground truth labels for datasets, such as the indices of the "shorcut samples" in the Shortcut Detection metric, or the mislabeling (noisy) label indices for the Mislabeling Detection Metric. However, users often may not have access to these labels. To address this, we recommend either using one of our pre-built benchmark suites or training (using the ``train`` method) a custom benchmark for comparing explainers. Benchmarks provide a controlled environment for systematic evaluation.

-  **Caching**: Many explainers in our library generate re-usable cache. The ``cache_id`` and ``model_id`` parameters passed to various class instances are used to store these intermediary results. Ensure each experiment is assigned a unique combination of these arguments. Failing to do so could lead to incorrect reuse of cached results. If you wish to avoid re-using cached results, you can set the ``load_from_disk`` parameter to ``False``.

-  **Explanations Are Expensive To Compute**: Certain explainers, such as TracInCPRandomProj, may lead to OutOfMemory (OOM) issues when applied to large models or datasets. In such cases, we recommend adjusting memory usage by either reducing the dataset size or using smaller models to avoid these issues.
