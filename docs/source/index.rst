

Welcome to quanda's documentation!
==================================

|quanda| is a toolkit for **quan**\ titative evaluation of **d**\ ata **a**\ ttribution methods in **PyTorch**.

.. note::
    |quanda| is currently in development. We are actively working on expanding the library and improving the documentation. If you have any questions, please `open an issue <https://github.com/dilyabareeva/quanda/issues/new/choose>`_ or write us at dilyabareeva@gmail.com or galip.uemit.yolcu@hhi.fraunhofer.de.


.. figure:: https://datacloud.hhi.fraunhofer.de/s/kkwrCaYHffW5J8R/preview
   :alt: Figure 1
   :align: left
   :width: 120%

   |quanda| provides a unified and standardized framework to evaluate the quality of Training Data Attribution methods in different contexts and from different perspectives.

.. note::
    This page describes |quanda|'s purpose, design and features. For a quick start on |quanda|, please refer to the :doc:`getting started <./getting_started>` page.

What is Training Data Attribution?
----------------------------------
The interpretability of neural network decisions is a prolific field which has seen a variety of approaches over time. Most of the initial focus was on feature attribution methods, which highlight features in the input space that are responsible for a specific prediction (`Simonyan et al., 2014 <https://arxiv.org/abs/1312.6034>`_; `Bach et al., 2015 <https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140>`_; `Lundberg and Lee, 2017 <https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf>`_). These methods were often criticized for being unreliable and difficult to understand (`Adebayo et al., 2018 <https://proceedings.neurips.cc/paper_files/paper/2018/file/294a8ed24b1ad22ec2e7efea049b8737-Paper.pdf>`_; `Ghorbani et al., 2019 <https://ojs.aaai.org/index.php/AAAI/article/view/4252>`_). In response, researchers explored new directions, such as concept-based (`Poeta et al., 2023 <https://arxiv.org/abs/2312.12936>`_) and mechanistic interpretability (`Bereska and Gavves <https://openreview.net/forum?id=ePUVetPKu6>`_) methods. Recently, **Training Data Attribution** (TDA) has gained attention as a promising approach to enhancing the interpretability of neural networks.

TDA methods attribute model output on a specific test sample to the training dataset that it was trained on. As such, they reveal the training datapoints responsible for the model's decisions. Tracing model decisions back to the training data, TDA methods enable practitioners to understand the model's behavior and identify potential issues in the training setup, e.g. biases in the dataset. Different approaches have been proposed for this problem. While some methods focus on estimating the counterfactual effect of removing datapoints from the training set and retraining the model (`Koh and Liang, 2017 <https://proceedings.mlr.press/v70/koh17a.html>`_; `Park et al., 2023 <https://proceedings.mlr.press/v202/park23c.html>`_; `Bae et al., 2024 <https://arxiv.org/abs/2405.12186>`_), other methods achieve the attribution by tracking the contributions of training points to the loss reduction throughout training (`Pruthi et al., 2020 <https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html>`_), using interpretable surrogate models (`Yeh et al., 2018 <https://proceedings.neurips.cc/paper/2018/hash/8a7129b8f3edd95b7d969dfc2c8e9d9d-Abstract.html>`_) or finding training samples that are deemed similar to the test sample by the model (`Caruana et. al, 1999 <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2232607/>`_; `Hanawa et. al, 2021 <https://openreview.net/forum?id=9uvhpyQwzM_>`_). In addition to model understanding, TDA has been used in a variety of applications such as debugging model behavior (`Koh and Liang, 2017 <https://proceedings.mlr.press/v70/koh17a.html>`_; `Yeh et al., 2018 <https://proceedings.neurips.cc/paper/2018/hash/8a7129b8f3edd95b7d969dfc2c8e9d9d-Abstract.html>`_; `K and Søgaard, 2021 <https://arxiv.org/abs/2111.04683>`_; `Guo et al., 2021 <https://aclanthology.org/2021.emnlp-main.808>`_), data summarization (`Khanna et al., 2019 <https://proceedings.mlr.press/v89/khanna19a.html>`_; `Marion et al., 2023 <https://openreview.net/forum?id=XUIYn3jo5T>`_; `Yang et al., 2023 <https://openreview.net/forum?id=4wZiAXD29TQ>`_), dataset selection (`Engstrom et al., 2024 <https://openreview.net/forum?id=GC8HkKeH8s>`_; `Chhabra et al., 2024 <https://openreview.net/forum?id=HE9eUQlAvo>`_), fact tracing (`Akyurek et al., 2022 <https://aclanthology.org/2022.findings-emnlp.180>`_) and machine unlearning (`Warnecke
et al., 2023 <https://arxiv.org/abs/2108.11577>`_).

How to Assess the Quality of Attributions?
------------------------------------------

Evaluation of interpretability approaches is a challenging task, as it is often difficult to define a ground truth for interpretability. Although there are various demonstrations of TDA’s potential for interpretability and practical applications, the critical question of how TDA methods should be effectively evaluated remains open. While methods based on estimating counterfactual retraining effects have a well defined ground truth, this ground truth is computationally demanding and is not feasibly computable for large scale experiments. To address these shortcomings, several approaches have been proposed by the community, which can be categorized into three groups:

- **Ground truth**: As some of the methods are designed to approximate LOO effects, ground truth can often be computed for TDA evaluation. As explained above, this counterfactual ground truth approach requires retraining the model multiple times on different subsets of the training data, which quickly becomes computationally expensive. Additionally, this ground truth is shown to be dominated by noise in practical deep learning settings, due to the inherent stochasticity of a typical training process (`Basu et al., 2021 <https://openreview.net/forum?id=xHKVVHGDOEk>`_; `Nguyen et al., 2023 <https://proceedings.neurips.cc/paper_files/paper/2023/hash/ca774047bc3b46cc81e53ead34cd5d5a-Abstract-Conference.html>`_). The most straightforward example of ground truth metrics is the Leave-one-out (LOO) metric (`Koh and Liang, 2017 <https://proceedings.mlr.press/v70/koh17a.html>`_). Linear Datamodeling Score (LDS) (`Park et al., 2023 <https://proceedings.mlr.press/v202/park23c.html>`_) is another example of a ground truth metric that measures the correlation between the (grouped) attribution scores and the actual output of models trained on a limited number subsets of the training set, which helps with the computational demand of the metric, but doesn't solve the problem in its totality.

- **Downstream Task Evaluators**: To remedy the challenges associated with ground truth evaluation, the literature proposes to assess the utility of a TDA method within the context of an end-task. The most commonly used evaluation criteria is Mislabeling Detection (`Koh and Liang, 2017 <https://proceedings.mlr.press/v70/koh17a.html>`_; `Yeh et al., 2018 <https://proceedings.neurips.cc/paper/2018/hash/8a7129b8f3edd95b7d969dfc2c8e9d9d-Abstract.html>`_; `Pruthi et al., 2020 <https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html>`_) which compares different TDA methods in terms of their usefulness for detecting mislabeled samples after training the network on a dataset of which the labels are deliberately poisoned. Other examples could be detecting backdoor attacks (`Karthikeyan et al., 2021 <https://arxiv.org/abs/2111.04683>`_; `Yolcu et al., 2024 <https://arxiv.org/abs/2402.12118>`_) or predicting the model decision from its attributions (`Hanawa et. al, 2021 <https://openreview.net/forum?id=9uvhpyQwzM_>`_).

- **Heuristics**: Finally, the community also made use of heuristics (desirable properties or sanity checks) to evaluate the quality of TDA techniques. These include comparing the attributions of a trained model and a randomized model (`Hanawa et. al, 2021 <https://openreview.net/forum?id=9uvhpyQwzM_>`_) and measuring the amount of overlap between the attributions for different test samples (`Barshan et al., 2020 <http://proceedings.mlr.press/v108/barshan20a/barshan20a.pdf>`_).

|quanda| is designed to meet the need of a comprehensive and systematic evaluation framework, allowing practitioners and researchers to obtain a detailed view of the performance of TDA methods in various contexts.

Library Features
----------------
Here we list the main components of |quanda| along with basic explanations of their function. We refer the reader to the :doc:`contribution guide <./contributing_to_quanda>`, the :doc:`API reference <docs_api/modules>` and the :ref:`Basic Usage <getting_started:basic usage>`. Below is a schematic representation of the components of |quanda|:

.. figure:: _static/components.png
   :alt: Figure 2
   :align: center
   :width: 90%

   Components and their interactions in |quanda|


- **Unified TDA Interface**: |quanda| provides a unified interface for various TDA methods, symbolized by the :doc:`Explainer <docs_api/quanda.explainers.base>` base class. The interface design prioritizes ease of use and easy extensions, allowing users to quickly wrap their implementations to use within |quanda|. This allows users to easily incorporate existing explainers and compare them against each other.
- **Metrics**: |quanda| provides a set of metrics to evaluate the effectiveness of TDA methods. These metrics are based on the latest research in the field. Most :doc:`Metric <docs_api/quanda.metrics.base>` objects in |quanda| are used to compute the evaluation scores from attributions over a test set. After initialization with all the necessary information, the user can call the `update` method to update the metric with the generated attributions. The `compute` method can then be called to compute the final evaluation score. The :doc:`Metric <docs_api/quanda.metrics.base>` objects are designed to be easily extendable, allowing users to define their own metrics.
- **Benchmarking**: Note that many metrics require training models in controlled settings, e.g. with mislabeled samples that are known. This means that the corresponding :doc:`Metric <docs_api/quanda.metrics.base>` objects can only be used if the user has prepared this controlled setup. Furthermore, :doc:`Metric <docs_api/quanda.metrics.base>` objects require generating the attributions beforehand. |quanda| provides a benchmarking tool to evaluate the performance of TDA methods on a given model, dataset and problem. For each :doc:`Metric <docs_api/quanda.metrics.base>` object, |quanda| provides a corresponding `Benchmark` object.  The `Benchmark` objects handle the creation of the controlled setup, training the model, generating the attributions and evaluating them using the corresponding :doc:`Metric <docs_api/quanda.metrics.base>` object, if needed. This is done via the use of the `generate` method to initialize the :doc:`Benchmark <docs_api/quanda.benchmarks.base>` object. After initialization, running the `evaluate` method will use the supplied :doc:`Explainer <docs_api/quanda.explainers.base>` class to generate explanations and the associated :doc:`Metric <docs_api/quanda.metrics.base>` object to compute the evaluation. In case you already have created a controlled setup, the `assemble` method can be used to generate a benchmark with user supplied assets. Finally, we provide precomputed benchmarks, which can be used by initializing the object with the `load` method. Theses precomputed benchmarks allow the user to skip the creation of the controlled setup to directly start the evaluation process, while providing a standard benchmark for practitioners and researchers to compare their methods with.

Supported TDA Methods
---------------------
.. list-table::
   :header-rows: 1

   * - Method
     - Repository
     - Reference
     - Description
   * - Similarity Influence
     - `Captum <https://github.com/pytorch/captum/tree/master>`_
     - `Caruana et al., 1999 <https://captum.ai/api/influence.html#similarityinfluence>`_
     - Ranks the training samples based on their similarity to the test sample
   * - Arnoldi Influence Functions
     - `Captum <https://github.com/pytorch/captum/tree/master>`_
     - `Schioppa et al., 2022 <https://arxiv.org/abs/2112.03052>`_
     -  Estimates LOO effects, following (`Koh and Liang, 2017 <https://proceedings.mlr.press/v70/koh17a.html>`_)
   * - TracIn
     - `Captum <https://github.com/pytorch/captum/tree/master>`_
     - `Pruthi et al., 2020 <https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html>`_
     - Tracks the contribution of training points in the loss reduction throughout training, via a linear approximation
   * - Representer Point Selection
     - `Representer Point Selection <https://github.com/chihkuanyeh/Representer_Point_Selection>`_
     - `Yeh et al., 2018 <https://proceedings.neurips.cc/paper/2018/hash/8a7129b8f3edd95b7d969dfc2c8e9d9d-Abstract.html>`_
     - Trains the model with L2 regularization on the final layer, which produces an interpretable surrogate model
   * - TRAK
     - `TRAK <https://github.com/MadryLab/trak>`_
     - `Park et al., 2023 <https://proceedings.mlr.press/v202/park23c.html>`_
     - Uses an empirical Neural Tangent Kernel surrogate model for which a theoretical TDA formula exists

Evaluation Metrics
------------------
In this section, we list the evaluation criteria that are currently available in |quanda|.


- **Linear Datamodeling Score** (`Park et al., 2023 <https://proceedings.mlr.press/v202/park23c.html>`_): Measures the correlation between the (grouped) attribution scores and the actual output of models trained on different subsets of the training set. For each subset, the linear datamodeling score compares the actual model output with the sum of attribution scores from the subset using Spearman rank correlation.

- **Identical Class / Identical Subclass** (`Hanawa et al., 2021 <https://openreview.net/forum?id=9uvhpyQwzM_>`_): Measures the proportion of identical classes or subclasses in the top-1 training samples over the test dataset. If the attributions are based on similarity, they are expected to be predictive of the class of the test datapoint, as well as different subclasses under a single label.

-  **Top-K Cardinality**  (`Barshan et al., 2020 <http://proceedings.mlr.press/v108/barshan20a/barshan20a.pdf>`_): Measures the cardinality of the union of the top-K training samples. Since the attributions are expected to be dependent on the test input, they are expected to vary heavily for different test points, resulting in a low overlap (high metric value).

- **Model Randomization** (`Hanawa et al., 2021 <https://openreview.net/forum?id=9uvhpyQwzM_>`_): Measures the correlation between the original TDA and the TDA of a model with randomized weights. Since the attributions are expected to depend on model parameters, the correlation between original and randomized attributions should be low.

- **Data Cleaning** (`Khanna et al., 2019 <https://proceedings.mlr.press/v89/khanna19a.html>`_): Uses TDA to identify training samples responsible for misclassification. Removing them from the training set, we expect to see an improvement in the model performance when we retrain the model.

- **Mislabeled Data Detection** (`Koh and Liang, 2017 <https://proceedings.mlr.press/v70/koh17a.html>`_): Computes the proportion of noisy training labels detected as a function of the percentage of inspected training samples. The samples are inspected in order according to their global TDA ranking, which is computed using local attributions. This produces a cumulative mislabeling detection curve. We expect to see a curve that rapidly increases as we check more of the training data, thus we compute the area under this curve.

Benchmarks
----------
|quanda| comes with a few pre-computed benchmarks that can be conveniently used for evaluation in a plug-and-play manner. We are planning to significantly expand the number of benchmarks in the future. The benchmarks currently use the MNIST dataset to conduct evaluations. The following benchmarks are available:

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

Citation
--------
If you find |quanda| useful and want to use it in your research, please cite it using the following BibTeX entry:

.. code::
  @misc{bareeva2024quandainterpretabilitytoolkittraining,
        title={Quanda: An Interpretability Toolkit for Training Data Attribution Evaluation and Beyond},
        author={Dilyara Bareeva and Galip Ümit Yolcu and Anna Hedström and Niklas Schmolenski and Thomas Wiegand and Wojciech Samek and Sebastian Lapuschkin},
        year={2024},
        eprint={2410.07158},
        archivePrefix={arXiv},
        primaryClass={cs.LG},
        url={https://arxiv.org/abs/2410.07158},
  }

.. toctree::
   :caption: Helpful Pages

   getting_started
   contributing_to_quanda
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
