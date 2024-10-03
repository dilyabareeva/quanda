.. quanda documentation master file, created by
   sphinx-quickstart on Wed Jul  3 23:40:28 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to quanda's documentation!
==================================

**quanda** is a toolkit for **quan**\ titative evaluation of **d**\ ata **a**\ ttribution methods in **PyTorch**.

**Training data attribution** (TDA) methods attribute model output to its training samples (`Koh and Liang, 2017 <https://proceedings.mlr.press/v70/koh17a.html>`__; `Yeh et al., 2018 <https://proceedings.neurips.cc/paper/2018/hash/8a7129b8f3edd95b7d969dfc2c8e9d9d-Abstract.html>`__; `Park et al., 2023 <https://proceedings.mlr.press/v202/park23c.html>`__; `Pruthi et al., 2020 <https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html>`__; `Bae et al., 2024 <https://arxiv.org/abs/2405.12186>`__). Outside of being used for understanding models, TDA has also found usage in a large variety of applications such as debugging model behavior (`Koh and Liang, 2017 <https://proceedings.mlr.press/v70/koh17a.html>`__; `Yeh et al., 2018 <https://proceedings.neurips.cc/paper/2018/hash/8a7129b8f3edd95b7d969dfc2c8e9d9d-Abstract.html>`__; `K and Søgaard, 2021 <https://arxiv.org/abs/2111.04683>`__; `Guo et al., 2021 <https://aclanthology.org/2021.emnlp-main.808>`__), data summarization (`Khanna et al., 2019 <https://proceedings.mlr.press/v89/khanna19a.html>`__; `Marion et al., 2023 <https://openreview.net/forum?id=XUIYn3jo5T>`__; `Yang et al., 2023 <https://openreview.net/forum?id=4wZiAXD29TQ>`__), dataset selection (`Engstrom et al., 2024 <https://openreview.net/forum?id=GC8HkKeH8s>`__; `Chhabra et al., 2024 <https://openreview.net/forum?id=HE9eUQlAvo>`__), fact tracing (`Akyurek et al., 2022 <https://aclanthology.org/2022.findings-emnlp.180>`__) and machine unlearning (`Warnecke et al., 2023 <https://arxiv.org/abs/2108.11577>`__).


The evaluation of TDA methods is a difficult task, especially due to the computationally demanding and noisy nature of the ground truths. (`Basu et al. <https://arxiv.org/abs/2006.14651>`__; `Nguyen et al. <https://arxiv.org/abs/2305.19765>`__). For this reason, the community has proposed various sanity checks (`Hanawa et al., 2021 <https://openreview.net/forum?id=9uvhpyQwzM_>`__) and downstream tasks (`Koh and Liang, 2017 <https://proceedings.mlr.press/v70/koh17a.html>`__; `Khanna et al., 2019 <https://proceedings.mlr.press/v89/khanna19a.html>`__; `Karthikeyan et al. <https://arxiv.org/abs/2111.04683>`__) to evaluate the effectiveness of TDA methods. Quanda provides a unified framework to evaluate TDA methods to help researchers and practitioners choose between the proposed methods.

Currently implemented in **quanda** are the following evaluation strategies:

Library Features
----------------

- **Unified TDA Interface**: **quanda** provides a unified interface for various TDA methods, allowing users to easily switch between different methods.
- **Metrics**: **quanda** provides a set of metrics to evaluate the effectiveness of TDA methods. These metrics are based on the latest research in the field.
- **Benchmarking**: **quanda** provides a benchmarking tool to evaluate the performance of TDA methods on a given model, dataset and problem. As many TDA evaluation methods require access to ground truth, our benchmarking tools allow to generate a controlled setting with ground truth, and then compare the performance of different TDA methods on this setting.

Supported TDA Methods
---------------------

+-----------------------------+---------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| Method                      | Repository                                                                                  | Reference                                                                                                              |
+=============================+=============================================================================================+========================================================================================================================+
| Similarity Influence        | `Captum <https://github.com/pytorch/captum/tree/master>`_                                   | `Caruana et al., 1999 <https://captum.ai/api/influence.html#similarityinfluence>`_                                     |
+-----------------------------+---------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| Arnoldi Influence Functions | `Captum <https://github.com/pytorch/captum/tree/master>`_                                   | `Schioppa et al., 2022 <https://arxiv.org/abs/2112.03052>`_                                                            |
+-----------------------------+---------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| TracIn                      | `Captum <https://github.com/pytorch/captum/tree/master>`_                                   | `Pruthi et al., 2020 <https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html>`_ |
+-----------------------------+---------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| Representer Point Selection | `Representer Point Selection <https://github.com/chihkuanyeh/Representer_Point_Selection>`_ | `Yeh et al., 2018 <https://proceedings.neurips.cc/paper/2018/hash/8a7129b8f3edd95b7d969dfc2c8e9d9d-Abstract.html>`_    |
+-----------------------------+---------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+
| TRAK                        | `TRAK <https://github.com/MadryLab/trak>`_                                                  | `Park et al., 2023 <https://proceedings.mlr.press/v202/park23c.html>`_                                                 |
+-----------------------------+---------------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------+


Evaluation Metrics
------------------
- **Linear Datamodeling Score** (`Park et al., 2023 <https://proceedings.mlr.press/v202/park23c.html>`_): Measures the correlation between the (grouped) attribution scores and the actual output of models trained on different subsets of the training set. For each subset, the linear datamodeling score compares the actual model output with the sum of attribution scores from the subset using Spearman rank correlation.

- **Identical Class / Identical Subclass** (`Hanawa et al., 2021 <https://openreview.net/forum?id=9uvhpyQwzM_>`_): Measures the proportion of identical classes or subclasses in the top-1 training samples over the test dataset. If the attributions are based on similarity, they are expected to be predictive of the class of the test datapoint, as well as different subclasses under a single label.

-  **Top-K Overlap**  (`Barshan et al., 2020 <http://proceedings.mlr.press/v108/barshan20a/barshan20a.pdf>`_): Measures the cardinality of the union of the top-K training samples. Since the attributions are expected to be dependent on the test input, they are expected to vary heavily for different test points, resulting in a low overlap (high metric value).

- **Model Randomization** (`Hanawa et al., 2021 <https://openreview.net/forum?id=9uvhpyQwzM_>`_): Measures the correlation between the original TDA and the TDA of a model with randomized weights. Since the attributions are expected to depend on model parameters, the correlation between original and randomized attributions should be low.

- **Data Cleaning** (`Khanna et al., 2019 <https://proceedings.mlr.press/v89/khanna19a.html>`_): Uses TDA to identify training samples responsible for misclassification. Removing them from the training set, we expect to see an improvement in the model performance when we retrain the model.

- **Mislabeled Data Detection** (`Koh and Liang, 2017 <https://proceedings.mlr.press/v70/koh17a.html>`_): Computes the proportion of noisy training labels detected as a function of the percentage of inspected training samples. The samples are inspected in order according to their global TDA ranking, which is computed using local attributions. This produces a cumulative mislabeling detection curve. We expect to see a curve that rapidly increases as we check more of the training data, thus we compute the area under this curve.


.. toctree::
   getting_started
   tutorials


.. toctree::
   :caption: API Reference
   :maxdepth: 1

   docs_api/modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
