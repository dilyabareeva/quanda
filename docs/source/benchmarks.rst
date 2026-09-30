Available Benchmarks
====================

This page lists all pre-computed benchmarks that come with |quanda|. Each ID is
passed to the ``load_pretrained`` method of the corresponding
:doc:`Benchmark <docs_api/quanda.benchmarks.base>` class, which downloads the
benchmark assets (model checkpoints, dataset splits and metadata) and prepares
the evaluation setting:

.. code-block:: python

   from quanda.benchmarks.ground_truth import LinearDatamodeling

   benchmark = LinearDatamodeling.load_pretrained(
       bench_id="cifar_alpha09_linear_datamodeling",
       cache_dir=cache_dir,
   )

The benchmark IDs are mapped to their configuration files in
``quanda/benchmarks/resources/config_map.py``. The configuration files
themselves live in ``quanda/benchmarks/resources/configs`` and document the
exact model, dataset, training and metric parameters of each benchmark.

.. warning::
    The LDS benchmark downloads 101 checkpoints (the main
    model plus every subset model) and evaluation runs a forward pass over the
    eval set for each of them. See the
    :doc:`Linear Datamodeling Score (LDS) <tutorial_pages/lds>` tutorial for
    the caveats and for how to precompute and reuse the counterfactual subset
    logits across explainers.

Vision
------

.. list-table::
  :header-rows: 1

  * - Benchmark ID
    - Metric
    - Dataset / Model
  * - mnist_class_detection
    - `ClassDetection <docs_api/quanda.metrics.downstream_eval.class_detection.html>`_
    - MNIST / LeNet
  * - mnist_subclass_detection
    - `SubclassDetection <docs_api/quanda.metrics.downstream_eval.subclass_detection.html>`_
    - MNIST / LeNet
  * - mnist_mislabeling_detection
    - `MislabelingDetection <docs_api/quanda.metrics.downstream_eval.mislabeling_detection.html>`_
    - MNIST / LeNet
  * - mnist_shortcut_detection
    - `ShortcutDetection <docs_api/quanda.metrics.downstream_eval.shortcut_detection.html>`_
    - MNIST / LeNet
  * - mnist_top_k_cardinality
    - `TopKCardinality <docs_api/quanda.metrics.heuristics.top_k_cardinality.html>`_
    - MNIST / LeNet
  * - mnist_model_randomization
    - `ModelRandomization <docs_api/quanda.metrics.heuristics.model_randomization.html>`_
    - MNIST / LeNet
  * - mnist_mixed_datasets
    - `MixedDatasets <docs_api/quanda.metrics.heuristics.mixed_datasets.html>`_
    - MNIST / LeNet
  * - mnist_linear_datamodeling
    - `LinearDatamodelingScore <docs_api/quanda.metrics.ground_truth.linear_datamodeling.html>`_
    - MNIST / LeNet
  * - cifar_class_detection
    - `ClassDetection <docs_api/quanda.metrics.downstream_eval.class_detection.html>`_
    - CIFAR-10 / ResNet-9
  * - cifar_subclass_detection
    - `SubclassDetection <docs_api/quanda.metrics.downstream_eval.subclass_detection.html>`_
    - CIFAR-10 / ResNet-9
  * - cifar_mislabeling_detection
    - `MislabelingDetection <docs_api/quanda.metrics.downstream_eval.mislabeling_detection.html>`_
    - CIFAR-10 / ResNet-9
  * - cifar_shortcut_detection
    - `ShortcutDetection <docs_api/quanda.metrics.downstream_eval.shortcut_detection.html>`_
    - CIFAR-10 / ResNet-9
  * - cifar_top_k_cardinality
    - `TopKCardinality <docs_api/quanda.metrics.heuristics.top_k_cardinality.html>`_
    - CIFAR-10 / ResNet-9
  * - cifar_model_randomization
    - `ModelRandomization <docs_api/quanda.metrics.heuristics.model_randomization.html>`_
    - CIFAR-10 / ResNet-9
  * - cifar_mixed_datasets
    - `MixedDatasets <docs_api/quanda.metrics.heuristics.mixed_datasets.html>`_
    - CIFAR-10 / ResNet-9
  * - cifar_linear_datamodeling
    - `LinearDatamodelingScore <docs_api/quanda.metrics.ground_truth.linear_datamodeling.html>`_
    - CIFAR-10 / ResNet-9
  * - awa2_class_detection
    - `ClassDetection <docs_api/quanda.metrics.downstream_eval.class_detection.html>`_
    - AWA2 / ResNet-50
  * - awa2_subclass_detection
    - `SubclassDetection <docs_api/quanda.metrics.downstream_eval.subclass_detection.html>`_
    - AWA2 / ResNet-50
  * - awa2_mislabeling_detection
    - `MislabelingDetection <docs_api/quanda.metrics.downstream_eval.mislabeling_detection.html>`_
    - AWA2 / ResNet-50
  * - awa2_shortcut_detection
    - `ShortcutDetection <docs_api/quanda.metrics.downstream_eval.shortcut_detection.html>`_
    - AWA2 / ResNet-50
  * - awa2_top_k_cardinality
    - `TopKCardinality <docs_api/quanda.metrics.heuristics.top_k_cardinality.html>`_
    - AWA2 / ResNet-50
  * - awa2_model_randomization
    - `ModelRandomization <docs_api/quanda.metrics.heuristics.model_randomization.html>`_
    - AWA2 / ResNet-50
  * - awa2_mixed_datasets
    - `MixedDatasets <docs_api/quanda.metrics.heuristics.mixed_datasets.html>`_
    - AWA2 / ResNet-50
  * - awa2_linear_datamodeling
    - `LinearDatamodelingScore <docs_api/quanda.metrics.ground_truth.linear_datamodeling.html>`_
    - AWA2 / ResNet-50

Text Classification
-------------------

.. list-table::
  :header-rows: 1

  * - Benchmark ID
    - Metric
    - Dataset / Model
  * - qnli_class_detection
    - `ClassDetection <docs_api/quanda.metrics.downstream_eval.class_detection.html>`_
    - QNLI / BERT
  * - qnli_mislabeling_detection
    - `MislabelingDetection <docs_api/quanda.metrics.downstream_eval.mislabeling_detection.html>`_
    - QNLI / BERT
  * - qnli_top_k_cardinality
    - `TopKCardinality <docs_api/quanda.metrics.heuristics.top_k_cardinality.html>`_
    - QNLI / BERT
  * - qnli_model_randomization
    - `ModelRandomization <docs_api/quanda.metrics.heuristics.model_randomization.html>`_
    - QNLI / BERT
  * - qnli_mixed_datasets
    - `MixedDatasets <docs_api/quanda.metrics.heuristics.mixed_datasets.html>`_
    - QNLI / BERT
  * - | qnli_linear_datamodeling
      | qnli_alpha075_linear_datamodeling
    - `LinearDatamodelingScore <docs_api/quanda.metrics.ground_truth.linear_datamodeling.html>`_
    - QNLI / BERT

Causal Language Modeling
------------------------

.. list-table::
  :header-rows: 1

  * - Benchmark ID
    - Metric
    - Dataset / Model
  * - gpt2_trex_openwebtext_ft_mrr
    - `MRR <docs_api/quanda.metrics.downstream_eval.mrr.html>`_
    - T-REx / GPT-2 fine-tuned on OpenWebText
  * - gpt2_trex_openwebtext_ft_recall_at_k
    - `RecallAtK <docs_api/quanda.metrics.downstream_eval.recall_at_k.html>`_
    - T-REx / GPT-2 fine-tuned on OpenWebText
  * - gpt2_trex_openwebtext_ft_tail_patch
    - `TailPatch <docs_api/quanda.metrics.downstream_eval.tail_patch.html>`_
    - T-REx / GPT-2 fine-tuned on OpenWebText

.. _lds-subset-size:

Linear Datamodeling Score: Subset Size Variants
-----------------------------------------------

The `LinearDatamodelingScore <docs_api/quanda.metrics.ground_truth.linear_datamodeling.html>`_
retrains the model on ``m`` random subsets of the training set, each containing
a fraction ``alpha`` of the training data, and correlates the attribution
scores summed over each subset with the output of the corresponding retrained
model. The
benchmarks below therefore cover a range of subset sizes; they all use
``m = 100`` retrained models and differ only in ``alpha``:

+---------------------+---------------+-------------------------------------+
| Setting             | ``alpha``     | Benchmark ID                        |
+=====================+===============+=====================================+
| MNIST / LeNet       | 0.5 (default) | mnist_linear_datamodeling           |
+                     +---------------+-------------------------------------+
|                     | 0.75          | mnist_alpha075_linear_datamodeling  |
+                     +---------------+-------------------------------------+
|                     | 0.9           | mnist_alpha09_linear_datamodeling   |
+                     +---------------+-------------------------------------+
|                     | 0.95          | mnist_alpha095_linear_datamodeling  |
+                     +---------------+-------------------------------------+
|                     | 0.999         | mnist_alpha0999_linear_datamodeling |
+---------------------+---------------+-------------------------------------+
| CIFAR-10 / ResNet-9 | 0.5 (default) | cifar_linear_datamodeling           |
+                     +---------------+-------------------------------------+
|                     | 0.75          | cifar_alpha075_linear_datamodeling  |
+                     +---------------+-------------------------------------+
|                     | 0.9           | cifar_alpha09_linear_datamodeling   |
+                     +---------------+-------------------------------------+
|                     | 0.95          | cifar_alpha095_linear_datamodeling  |
+                     +---------------+-------------------------------------+
|                     | 0.999         | cifar_alpha0999_linear_datamodeling |
+---------------------+---------------+-------------------------------------+
| AWA2 / ResNet-50    | 0.5 (default) | awa2_linear_datamodeling            |
+                     +---------------+-------------------------------------+
|                     | 0.75          | awa2_alpha075_linear_datamodeling   |
+---------------------+---------------+-------------------------------------+
| QNLI / BERT         | 0.5 (default) | qnli_linear_datamodeling            |
+                     +---------------+-------------------------------------+
|                     | 0.75          | qnli_alpha075_linear_datamodeling   |
+---------------------+---------------+-------------------------------------+


.. toctree::
   :maxdepth: 1

   benchmark_pages/validity_checks

