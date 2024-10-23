.. quanda documentation master file, created by
   sphinx-quickstart on Wed Jul  3 23:40:28 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to quanda's documentation!
==================================

**quanda** is a toolkit for **quan**\ titative evaluation of **d**\ ata **a**\ ttribution methods in **PyTorch**.

**Training data attribution** (TDA) methods attribute model output on a specific test sample to the training dataset that it was trained on. They reveal the training datapoints responsible for the model's decisions. Existing methods achieve this by estimating the counterfactual effect of removing datapoints from the training set (`Koh and Liang, 2017 <https://proceedings.mlr.press/v70/koh17a.html>`_; `Park et al., 2023 <https://proceedings.mlr.press/v202/park23c.html>`_; `Bae et al., 2024 <https://arxiv.org/abs/2405.12186>`_) tracking the contributions of training points to the loss reduction throughout training (`Pruthi et al., 2020 <https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html>`_), using interpretable surrogate models (`Yeh et al., 2018 <https://proceedings.neurips.cc/paper/2018/hash/8a7129b8f3edd95b7d969dfc2c8e9d9d-Abstract.html>`_) or finding training samples that are deemed similar to the test sample by the model (`Caruana et. al, 1999 <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2232607/>`_; `Hanawa et. al, 2021 <https://openreview.net/forum?id=9uvhpyQwzM_>`_). In addition to model understanding, TDA has been used in a variety of applications such as debugging model behavior (`Koh and Liang, 2017 <https://proceedings.mlr.press/v70/koh17a.html>`_; `Yeh et al., 2018 <https://proceedings.neurips.cc/paper/2018/hash/8a7129b8f3edd95b7d969dfc2c8e9d9d-Abstract.html>`_; `K and Søgaard, 2021 <https://arxiv.org/abs/2111.04683>`_; `Guo et al., 2021 <https://aclanthology.org/2021.emnlp-main.808>`_), data summarization (`Khanna et al., 2019 <https://proceedings.mlr.press/v89/khanna19a.html>`_; `Marion et al., 2023 <https://openreview.net/forum?id=XUIYn3jo5T>`_; `Yang et al., 2023 <https://openreview.net/forum?id=4wZiAXD29TQ>`_), dataset selection (`Engstrom et al., 2024 <https://openreview.net/forum?id=GC8HkKeH8s>`_; `Chhabra et al., 2024 <https://openreview.net/forum?id=HE9eUQlAvo>`_), fact tracing (`Akyurek et al., 2022 <https://aclanthology.org/2022.findings-emnlp.180>`_) and machine unlearning (`Warnecke
et al., 2023 <https://arxiv.org/abs/2108.11577>`_).

Although there are various demonstrations of TDA’s potential for interpretability and practical applications, the critical question of how TDA methods should be effectively evaluated remains open. Several approaches have been proposed by the community, which can be categorized into three groups:

- **Ground truth**: As some of the methods are designed to approximate LOO effects, ground truth can often be computed for TDA evaluation. However, this counterfactual ground truth approach requires retraining the model multiple times on different subsets of the training data, which quickly becomes computationally expensive. Additionally, this ground truth is shown to be dominated by noise in practical deep learning settings, due to the inherent stochasticity of a typical training process (`Basu et al., 2021 <https://openreview.net/forum?id=xHKVVHGDOEk>`_; `Nguyen et al., 2023 <https://proceedings.neurips.cc/paper_files/paper/2023/hash/ca774047bc3b46cc81e53ead34cd5d5a-Abstract-Conference.html>`_).

- **Downstream Task Evaluators**: To remedy the challenges associated with ground truth evaluation, the literature proposes to assess the utility of a TDA method within the context of an end-task, such as model debugging or data selection (`Koh and Liang, 2017 <https://proceedings.mlr.press/v70/koh17a.html>`_; `Khanna et al., 2019 <https://proceedings.mlr.press/v89/khanna19a.html>`_; `Karthikeyan et al., 2021 <https://arxiv.org/abs/2111.04683>`_)

- **Heuristics**: Finally, the community also used heuristics (desirable properties or sanity checks) to evaluate the quality of TDA techniques. These include comparing the attributions of a trained model and a randomized model (`Hanawa et. al, 2021 <https://openreview.net/forum?id=9uvhpyQwzM_>`_) and measuring the amount of overlap between the attributions for different test samples (`Barshan et al., 2020 <http://proceedings.mlr.press/v108/barshan20a/barshan20a.pdf>`_).

**quanda** is designed to meet the need of a comprehensive and systematic evaluation framework, allowing practitioners and researchers to obtain a detailed view of the performance of TDA methods in various contexts.

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

-  **Top-K Cardinality**  (`Barshan et al., 2020 <http://proceedings.mlr.press/v108/barshan20a/barshan20a.pdf>`_): Measures the cardinality of the union of the top-K training samples. Since the attributions are expected to be dependent on the test input, they are expected to vary heavily for different test points, resulting in a low overlap (high metric value).

- **Model Randomization** (`Hanawa et al., 2021 <https://openreview.net/forum?id=9uvhpyQwzM_>`_): Measures the correlation between the original TDA and the TDA of a model with randomized weights. Since the attributions are expected to depend on model parameters, the correlation between original and randomized attributions should be low.

- **Data Cleaning** (`Khanna et al., 2019 <https://proceedings.mlr.press/v89/khanna19a.html>`_): Uses TDA to identify training samples responsible for misclassification. Removing them from the training set, we expect to see an improvement in the model performance when we retrain the model.

- **Mislabeled Data Detection** (`Koh and Liang, 2017 <https://proceedings.mlr.press/v70/koh17a.html>`_): Computes the proportion of noisy training labels detected as a function of the percentage of inspected training samples. The samples are inspected in order according to their global TDA ranking, which is computed using local attributions. This produces a cumulative mislabeling detection curve. We expect to see a curve that rapidly increases as we check more of the training data, thus we compute the area under this curve.

Benchmarks
----------
**quanda** comes with a few pre-computed benchmarks that can be conveniently used for evaluation in a plug-and-play manner. We are planning to significantly expand the number of benchmarks in the future. The benchmarks currently use the MNIST dataset to conduct evaluations. The following benchmarks are available:

+---------------------------------+----------------------------------------------------------------------------------------------------+---------------------------+
|              Name               |                                               Metric                                               |           Type            |
+=================================+====================================================================================================+===========================+
|     mnist_top_k_cardinality     |        `TopKCardinalityMetric <docs_api/quanda.metrics.heuristics.top_k_cardinality.html>`_        |         Heuristic         |
+---------------------------------+----------------------------------------------------------------------------------------------------+---------------------------+
|      mnist_mixed_datasets       |           `MixedDatasetMetric <docs_api/quanda.metrics.heuristics.mixed_datasets.html>`_           |         Heuristic         |
+---------------------------------+----------------------------------------------------------------------------------------------------+---------------------------+
|      mnist_class_detection      |       `ClassDetectionMetric <docs_api/quanda.metrics.downstream_eval.class_detection.html>`_       | Downstream-Task-Evaluator |
+---------------------------------+----------------------------------------------------------------------------------------------------+---------------------------+
|    mnist_subclass_detection     |    `SubclassDetectionMetric <docs_api/quanda.metrics.downstream_eval.subclass_detection.html>`_    | Downstream-Task-Evaluator |
+---------------------------------+----------------------------------------------------------------------------------------------------+---------------------------+
|   mnist_mislabeling_detection   | `MislabelingDetectionMetric <docs_api/quanda.metrics.downstream_eval.mislabeling_detection.html>`_ | Downstream-Task-Evaluator |
+---------------------------------+----------------------------------------------------------------------------------------------------+---------------------------+
|    mnist_shortcut_detection     |    `ShortcutDetectionMetric <docs_api/quanda.metrics.downstream_eval.shortcut_detection.html>`_    | Downstream-Task-Evaluator |
+---------------------------------+----------------------------------------------------------------------------------------------------+---------------------------+
| mnist_linear_datamodeling_score |    `LinearDatamodelingMetric <docs_api/quanda.metrics.ground_truth.linear_datamodeling.html>`_     |       Ground Truth        |
+---------------------------------+----------------------------------------------------------------------------------------------------+---------------------------+


.. toctree::
   getting_started


.. toctree::
   :caption: API Reference
   :maxdepth: 1

   docs_api/modules
