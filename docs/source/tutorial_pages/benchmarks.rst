Benchmarks Tutorial
===================

Welcome to the benchmark tutorial of |quanda|. This tutorial walks you through the process of using the benchmarking tools in |quanda| to evaluate a data attribution method. This tutorial covers 3 different examples of benchmarks. It includes all different initialization schemes: training a benchmark from scratch using ``train()``, loading a benchmark from a YAML configuration using ``from_config()``, and downloading a precomputed benchmark using ``load_pretrained()``.

.. seealso::

   The :doc:`Linear Datamodeling Score (LDS) <lds>` page covers caveats specific to the most expensive benchmark in |quanda|, including how to precompute and reuse counterfactual subset logits across explainers.

To install the library with tutorial dependencies, run:

.. code:: bash

   pip install -e '.[tutorials]'

.. note::

   This tutorial is also available as a `notebook <https://github.com/dilyabareeva/quanda/blob/main/tutorials/demo_benchmarks.ipynb>`_.

Throughout this tutorial, we will be using a LeNet model trained on the MNIST dataset. Let's start the tutorial by importing the necessary libraries and components:

.. literalinclude:: ../../../tests/integration/test_benchmark_integration.py
   :language: python
   :start-after: # START1
   :end-before: # END1
   :dedent:

Next, we need to prepare for the following computations.

.. literalinclude:: ../../../tests/integration/test_benchmark_integration.py
   :language: python
   :start-after: # START2
   :end-before: # END2
   :dedent:

Downloading Precomputed Benchmarks
----------------------------------
In this part of the tutorial, we will use the :doc:`ShortcutDetection <../docs_api/quanda.benchmarks.downstream_eval.shortcut_detection>` metric.

We will use the benchmark corresponding to this metric to evaluate all data attributors currently included in |quanda| in terms of their ability to detect when the model is using a shortcut.

We will download the precomputed MNIST benchmark. This includes an MNIST dataset which has shortcut features (an 8-by-8 white box on a specific location) on a subset of its samples from the class 0, and a model trained on this dataset. This model has learned to classify images with these features as class 0, and we will measure the extent to which this is reflected in the attributions of different methods.

.. literalinclude:: ../../../tests/integration/test_benchmark_integration.py
   :language: python
   :start-after: # START3
   :end-before: # END3
   :dedent:

The benchmark object contains all information about the controlled evaluation setup. Run the following to get some samples with the shortcut features, using ``benchmark.train_dataset`` and ``benchmark.train_dataset.transform_indices``.

.. literalinclude:: ../../../tests/integration/test_benchmark_integration.py
   :language: python
   :start-after: # START4
   :end-before: # END4
   :dedent:

Prepare initialization parameters for TDA methods
+++++++++++++++++++++++++++++++++++++++++++++++++++
We now prepare the initialization parameters of attributors: hyperparameters, and components from the benchmark as needed. Note that we do not provide the model and dataset to use for attribution, since those components will be supplied by the benchmark objects, while initializing the attributor during evaluation.

- **Similarity Influence**:

.. literalinclude:: ../../../tests/integration/test_benchmark_integration.py
   :language: python
   :start-after: # START5
   :end-before: # END5
   :dedent:

- **Arnoldi Influence Functions**: Notice that the trained checkpoints have been saved to the ``cache_dir`` while downloading the benchmark. The checkpoint paths are available via ``benchmark.checkpoints``.

.. literalinclude:: ../../../tests/integration/test_benchmark_integration.py
   :language: python
   :start-after: # START6
   :end-before: # END6
   :dedent:

- **TracInCP**:

.. literalinclude:: ../../../tests/integration/test_benchmark_integration.py
   :language: python
   :start-after: # START7
   :end-before: # END7
   :dedent:

- **TRAK**:

.. literalinclude:: ../../../tests/integration/test_benchmark_integration.py
   :language: python
   :start-after: # START8
   :end-before: # END8
   :dedent:

- **Representer Point Selection**:

.. literalinclude:: ../../../tests/integration/test_benchmark_integration.py
   :language: python
   :start-after: # START9
   :end-before: # END9
   :dedent:

Run the benchmark evaluation on the attributors
+++++++++++++++++++++++++++++++++++++++++++++++
Note that some attributors take a long time to initialize or compute attributions. For a proof of concept, we recommend using :doc:`CaptumSimilarity <../docs_api/quanda.explainers.wrappers.captum_influence>` or :doc:`RepresenterPoints <../docs_api/quanda.explainers.wrappers.representer_points>`, or lowering the parameter values given above (i.e. using low ``proj_dim`` for TRAK or a low Hessian dataset size for :doc:`ArnoldiInfluence <../docs_api/quanda.explainers.wrappers.captum_influence>`)

.. literalinclude:: ../../../tests/integration/test_benchmark_integration.py
   :language: python
   :start-after: # START10
   :end-before: # END10
   :dedent:

At this point, the dictionary ``results`` contains the scores of the attributors on the benchmark.

Training a Benchmark from Scratch
-----------------------------------
We will now showcase how a benchmark can be created from a YAML configuration and trained from scratch. Quanda parses the configuration, sets up dataset manipulations, and trains the model. Then the benchmark can be used to evaluate different attributors. This is done through the ``Benchmark.train`` method.

We will go through this use-case with the :doc:`SubclassDetection <../docs_api/quanda.benchmarks.downstream_eval.subclass_detection>` benchmark which groups classes of the base dataset into superclasses. A model is trained to predict these superclasses, and the original label of the highest attributed datapoint for each test sample is observed. The benchmark expects this to be the same as the true class of the test sample.

The YAML configuration file specifies all required components:

- the model architecture and its training parameters (optimizer, scheduler, number of epochs, etc.)
- the training, validation, and evaluation datasets with their transforms
- a dataset wrapper (``LabelGroupingDataset``) that handles class grouping into superclasses
- the number of superclasses and the grouping strategy (``random`` or a specific mapping)

The class grouping can be set to ``random`` in the configuration to randomly assign classes into superclasses, which is the approach we will take in this tutorial.

.. important::

    The configuration must specify ``bench_save_dir``: the directory under which the trained benchmark (model checkpoints and metadata) is saved. There should be enough disk space to save the main model and M subset models for LDS (if applicable) under this directory. If training multiple benchmarks from scratch, make sure to set different ``bench_save_dir`` for each to avoid overwriting.

    If multiple training jobs must share a ``bench_save_dir`` (e.g. concurrent runs of the same benchmark), pass ``use_pid=True`` to ``train`` (or ``train_and_push_to_hub``) to suffix checkpoint and metadata directories with the current process id and avoid clobbering each other's outputs. By default ``use_pid=False``.

.. note::

    Please note that calling ``SubclassDetection.train`` will initiate model training, therefore it will potentially take a long time.

.. literalinclude:: ../../../tests/integration/test_benchmark_integration.py
   :language: python
   :start-after: # START14_1
   :end-before: # END14_1
   :dedent:

.. literalinclude:: ../../../tests/integration/test_benchmark_integration.py
   :language: python
   :start-after: # START14_2
   :end-before: # END14_2
   :dedent:

.. literalinclude:: ../../../tests/integration/test_benchmark_integration.py
   :language: python
   :start-after: # START14_3
   :end-before: # END14_3
   :dedent:

Now that we have trained the model on the MNIST dataset with grouped classes as defined in the configuration, we finalize this tutorial by evaluating the :doc:`CaptumSimilarity <../docs_api/quanda.explainers.wrappers.captum_influence>` attributor. The ``results`` dictionary will contain the score of the attributor on the benchmark after running the following:

.. literalinclude:: ../../../tests/integration/test_benchmark_integration.py
   :language: python
   :start-after: # START15
   :end-before: # END15
   :dedent:

Caching and Sharing Explanations
--------------------------------
Computing attributions is typically the most expensive step in TDA evaluation. To avoid recomputing them every time, every ``Benchmark`` exposes an ``explain`` classmethod that precomputes attributions over the evaluation dataset and writes them to disk together with an ``explanations_config.yaml`` describing how they were generated. A subsequent call to ``benchmark.evaluate(..., cache_dir=<path>, use_cached_expl=True)`` reads from that cache instead of recomputing.

The cache directory is keyed on:

- the benchmark id (or its ``explanations_group``, see below),
- the explainer class name,
- a stable hash of ``expl_kwargs``,
- the eval-subsample parameters ``max_eval_n`` and ``eval_seed``.

Changing any of these produces a different cache key, so cached explanations stay coupled to the exact setup they were computed on.

**Sharing across benchmarks.** Several benchmarks (e.g. ``ClassDetection`` and ``LinearDatamodelingMetric``) can be defined on top of the same model + train/eval datasets. Setting a common ``explanations_group`` in their YAML configs replaces the per-benchmark id segment of the cache key with a shared one, so a single attribution pass can drive multiple evaluations. Only opt in when the grouped benchmarks really share those inputs — mismatched inputs under a shared group will silently corrupt results.

See :doc:`lds` for the analogous mechanism that caches counterfactual subset logits across explainers in the LDS benchmark.
