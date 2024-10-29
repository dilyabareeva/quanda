Tutorials
=========
We have included a few tutorials to demonstrate the usage of |quanda|. To install the library with tutorial dependencies, run:

.. code:: bash

   pip install -e '.[tutorials]'

The tutorials currently included in |quanda| are:

- `Explainers <https://github.com/dilyabareeva/quanda/blob/main/tutorials/demo_explainers.ipynb>`_: shows how different explainers can be used with |quanda|. This tutorial goes through all the explainers that are included in |quanda| and walks through the steps of initializingenerating explanations and plotting them.
- `Metrics <https://github.com/dilyabareeva/quanda/blob/main/tutorials/demo_metrics.ipynb>`_: shows how to use the metrics in |quanda| to evaluate the performance of a method. This tutorial goes through all the metrics that are included in |quanda| and walks through the steps of initializing the metric and evaluating the performance of a method.
- :doc:`Benchmarks <tutorial_pages/benchmarks>`: shows how to use the benchmarking tools in |quanda| to evaluate a data attribution method. This tutorial includes 3 different examples of benchmarks. It includes all different initialization schemes: generating the benchmark from scratch, assembling a benchmark from existing assets and downloading a precomputed benchmark.

.. warning::

   The `Explainers` and `Metrics` tutorials depend on a data preparation and model training stage. In order to be able to run these tutorials, you need to go through this prerequisite step. Please run the `tutorial preparation notebook <https://github.com/dilyabareeva/quanda/blob/main/tutorials/demo_prep.ipynb>`_ before running these tutorials.
