
Welcome to |quanda|'s documentation!
====================================
|quanda| is a toolkit for **quan**\ titative evaluation of **d**\ ata **a**\ ttribution methods in **PyTorch**.

.. note::
    |quanda| is currently in development. We are actively working on expanding the library and improving the documentation. If you have any questions, please open an issue or write us an e-mail.

.. figure:: _static/fig_1_source.png
   :alt: Figure 1
   :align: left
   :width: 120%
   :figclass: only-light

   `Fig. 1:` |quanda| `provides a unified and standardized framework to evaluate the quality of Training Data Attribution methods in different contexts and from different perspectives.`

.. figure:: _static/fig_1_docs_darkmode.png
   :alt: Figure 1
   :align: left
   :width: 120%
   :figclass: only-dark

   `Fig. 1:` |quanda| `provides a unified and standardized framework to evaluate the quality of Training Data Attribution methods in different contexts and from different perspectives.`

.. note::
    This page describes |quanda|'s purpose, design and features. For a quick start on |quanda|, please refer to the :doc:`Quickstart <./quickstart>` page.

Training Data Attribution (TDA) is a new avenue in the interpretation of neural networks. While some methods attempt to estimate the counterfactual effects of training new models on the subsets of the training dataset, this ground truth is noisy and hard to compute. Therefore, the community has proposed evaluating these methods' performance on a downstream task, or measuring how well a method satisfies desired heuristic properties.

|quanda| is designed to meet the need of a comprehensive and systematic evaluation framework, as well as a unified interface for attributors. Please visit the :doc:`Background <./background>` page for a detailed explanation of TDA, including citations to the relevant literature.

Library Features
----------------
Here we list the main components of |quanda| along with basic explanations of their function. We refer the reader to the :doc:`Contribution guide <./contributing>`, the :doc:`API reference <docs_api/modules>` and the :doc:`Basic Usage <./quickstart>` for further details. Below is a schematic representation of the components of |quanda|:

.. figure:: _static/components.png
   :alt: Figure 2
   :align: center
   :width: 90%
   :figclass: only-light

   `Fig. 2: Components and their interactions in` |quanda|

.. figure:: _static/components-darkmode.png
   :alt: Figure 2
   :align: center
   :width: 90%
   :figclass: only-dark

   `Fig. 2: Components and their interactions in` |quanda|

.. raw:: html

   <details><summary><b><big>Explainers</big></b></summary>

|quanda| provides a unified interface for various TDA methods, symbolized by the :doc:`Explainer <docs_api/quanda.explainers.base>` base class. The interface design prioritizes ease of use and easy extensions, allowing users to quickly wrap their implementations to use within |quanda|.

.. raw:: html

   </details>

.. raw:: html

   <details><summary><b><big>Metrics</big></b></summary>

|quanda| provides a set of metrics to evaluate the effectiveness of TDA methods. These metrics are based on the latest research in the field. Most :doc:`Metric <docs_api/quanda.metrics.base>` objects in |quanda| are used to compute the evaluation scores from attributions over a test set. The :doc:`Metric <docs_api/quanda.metrics.base>` objects are designed to be easily extendable, allowing users to define their own metrics.

.. raw:: html

   </details>

.. raw:: html

   <details><summary><b><big>Benchmarks</big></b></summary>

Note that many metrics require training models in controlled settings, e.g. with mislabeled samples that are known. This means that the corresponding :doc:`Metric <docs_api/quanda.metrics.base>` objects can only be used if the user has prepared this controlled setup. Furthermore, :doc:`Metric <docs_api/quanda.metrics.base>` objects require generating the attributions beforehand. |quanda| provides a benchmarking tool to evaluate the performance of TDA methods on a given model, dataset and problem. For each :doc:`Metric <docs_api/quanda.metrics.base>` object, |quanda| provides a :doc:`Benchmark <docs_api/quanda.benchmarks.base>` object. The :doc:`Benchmark <docs_api/quanda.benchmarks.base>` objects handle the creation of the controlled setup, training the model, generating the attributions and evaluating them using the corresponding :doc:`Metric <docs_api/quanda.metrics.base>` object, if needed. Finally, we provide precomputed benchmarks, which can be used by initializing the object with the ``load`` method. These precomputed benchmarks allow the user to skip the creation of the controlled setup to directly start the evaluation process, while providing a standard benchmark for practitioners and researchers to compare their methods with.

.. raw:: html

   </details>

Supported TDA Libraries
-----------------------
.. list-table::
   :header-rows: 1

   * - Library
     - Reference
   * - `Captum <https://github.com/pytorch/captum/tree/master>`_ (Similarity Influence, Arnoldi Influence Functions, TracIn)
     - `Caruana et al., 1999 <https://captum.ai/api/influence.html#similarityinfluence>`_; `Schioppa et al., 2022 <https://arxiv.org/abs/2112.03052>`_; `Koh and Liang, 2017 <https://proceedings.mlr.press/v70/koh17a.html>`_; `Pruthi et al., 2020 <https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html>`_
   * - `Representer Point Selection <https://github.com/chihkuanyeh/Representer_Point_Selection>`_ (Representer Point Selection)
     - `Yeh et al., 2018 <https://proceedings.neurips.cc/paper/2018/hash/8a7129b8f3edd95b7d969dfc2c8e9d9d-Abstract.html>`_
   * - `TRAK <https://github.com/MadryLab/trak>`_ (TRAK)
     - `Park et al., 2023 <https://proceedings.mlr.press/v202/park23c.html>`_
   * - `Kronfluence <https://github.com/pomonam/kronfluence>`_ (Kronfluence)
     - `Grosse et al., 2023 <https://arxiv.org/abs/2308.03296>`_
   * - `Dattri <https://github.com/TRAIS-Lab/dattri>`_ (Influence Functions: Explicit / CG / LiSSA / DataInf, Arnoldi, EK-FAC, TracInCP, Grad-Dot, Grad-Cos, TRAK)
     - `Deng et al., 2024 <https://arxiv.org/abs/2410.04555>`_

Evaluation Metrics
------------------
In this section, we list the evaluation criteria that are currently available in |quanda|.

.. list-table::
  :header-rows: 1

  * - Name
    - Reference
    - Description
    - Type
  * - Linear Datamodeling Score
    - `Park et al., 2023 <https://proceedings.mlr.press/v202/park23c.html>`_
    - Measures the correlation between the (grouped) attribution scores and the actual output of models trained on different subsets of the training set. For each subset, the linear datamodeling score compares the actual model output with the sum of attribution scores from the subset using Spearman rank correlation.
    - Ground Truth
  * - Class Detection / Subclass Detection
    - `Hanawa et al., 2021 <https://openreview.net/forum?id=9uvhpyQwzM_>`_
    - Measures the proportion of identical classes or subclasses in the top-1 training samples over the test dataset. If the attributions are based on similarity, they are expected to be predictive of the class of the test datapoint, as well as different subclasses under a single label.
    - Downstream Task Evaluator
  * - Shortcut Detection
    - `Yolcu et al., 2024 <https://arxiv.org/abs/2402.12118>`_
    - Assuming a known `shortcut <https://www.nature.com/articles/s42256-020-00257-z>`_, or `Clever-Hans <https://www.nature.com/articles/s41467-019-08987-4>`_ effect has been identified in the model, this metric evaluates how effectively a TDA method can identify shortcut samples as the most influential in predicting cases with the shortcut artifact. This process is referred to as **Domain Mismatch Debugging** in the original paper.
    - Downstream Task Evaluator
  * - Mislabeled Data Detection
    - `Koh and Liang, 2017 <https://proceedings.mlr.press/v70/koh17a.html>`_
    - Computes the proportion of noisy training labels detected as a function of the percentage of inspected training samples. The samples are inspected in order according to their global TDA ranking, which is computed using local attributions. This produces a cumulative mislabeling detection curve. We expect to see a curve that rapidly increases as we check more of the training data, thus we compute the area under this curve.
    - Downstream Task Evaluator
  * - Top-K Cardinality
    - `Barshan et al., 2020 <http://proceedings.mlr.press/v108/barshan20a/barshan20a.pdf>`_
    - Measures the cardinality of the union of the top-K training samples. Since the attributions are expected to be dependent on the test input, they are expected to vary heavily for different test points, resulting in a low overlap (high metric value).
    - Heuristic
  * - Model Randomization
    - `Hanawa et al., 2021 <https://openreview.net/forum?id=9uvhpyQwzM_>`_
    - Measures the correlation between the original TDA and the TDA of a model with randomized weights. Since the attributions are expected to depend on model parameters, the correlation between original and randomized attributions should be low.
    - Heuristic
  * - Mixed Datasets
    - `Hammoudeh and Lowd, 2022 <https://dl.acm.org/doi/abs/10.1145/3548606.3559335>`_
    - In a setting where a model has been trained on two datasets: a clean dataset (e.g. CIFAR-10) and an adversarial (e.g. zeros from MNIST), this metric evaluates how well the model ranks the importance (attribution) of adversarial samples compared to clean samples when making predictions on an adversarial example.
    - Heuristic
  * - Mean Reciprocal Rank (MRR)
    - `Chang et al., 2024 <https://openreview.net/forum?id=KIPJKST4gw>`_
    - For fact-tracing settings, measures the mean reciprocal rank of the highest-ranked entailing proponent across fact queries.
    - Downstream Task Evaluator
  * - Recall@k
    - `Chang et al., 2024 <https://openreview.net/forum?id=KIPJKST4gw>`_
    - For fact-tracing settings, measures the proportion of facts for which an entailing proponent appears in the top-k retrievals.
    - Downstream Task Evaluator
  * - Tail Patch
    - `Chang et al., 2024 <https://openreview.net/forum?id=KIPJKST4gw>`_
    - For fact-tracing settings, measures the incremental change in target-sequence probability after taking a single training step on retrieved proponents.
    - Downstream Task Evaluator

Metric Interpretation Guideline
+++++++++++++++++++++++++++++++

.. list-table::
  :header-rows: 1

  * - Metric
    - Output range
    - Better
  * - `ClassDetection <docs_api/quanda.metrics.downstream_eval.class_detection.html>`_
    - ``[0, 1]``
    - higher
  * - `SubclassDetection <docs_api/quanda.metrics.downstream_eval.subclass_detection.html>`_
    - ``[0, 1]``
    - higher
  * - `MislabelingDetection <docs_api/quanda.metrics.downstream_eval.mislabeling_detection.html>`_
    - ``[0, 1]``
    - higher
  * - `ShortcutDetection <docs_api/quanda.metrics.downstream_eval.shortcut_detection.html>`_
    - ``[0, 1]``
    - higher
  * - `MixedDatasets <docs_api/quanda.metrics.heuristics.mixed_datasets.html>`_
    - ``[0, 1]``
    - higher
  * - `TopKCardinality <docs_api/quanda.metrics.heuristics.top_k_cardinality.html>`_
    - ``[0, 1]``
    - higher
  * - `ModelRandomization <docs_api/quanda.metrics.heuristics.model_randomization.html>`_
    - ``[-1, 1]``
    - closer to 0
  * - `LinearDatamodelingScore <docs_api/quanda.metrics.ground_truth.linear_datamodeling.html>`_
    - ``[-1, 1]``
    - higher
  * - `MRR <docs_api/quanda.metrics.downstream_eval.mrr.html>`_
    - ``[0, 1]``
    - higher
  * - `RecallAtK <docs_api/quanda.metrics.downstream_eval.recall_at_k.html>`_
    - ``[0, 1]``
    - higher
  * - `TailPatch <docs_api/quanda.metrics.downstream_eval.tail_patch.html>`_
    - ``[-1, 1]``
    - higher

Benchmarks
----------
|quanda| comes with a number of pre-computed benchmarks that can be conveniently used for evaluation in a plug-and-play manner. We are planning to significantly expand the number of benchmarks in the future. Currently available benchmarks span vision (MNIST / LeNet, CIFAR-10 / ResNet-9, AWA2 / ResNet-50), text classification (QNLI / BERT), and causal language modeling (T-REx / GPT-2 fine-tuned on OpenWebText).

.. list-table::
  :header-rows: 1

  * - Metric
    - Type
    - Modality
    - Benchmarks (Dataset / Model)
  * - `TopKCardinalityMetric <docs_api/quanda.metrics.heuristics.top_k_cardinality.html>`_
    - Heuristic
    - Vision
    - mnist_top_k_cardinality (MNIST / LeNet), cifar_top_k_cardinality (CIFAR-10 / ResNet-9), awa2_top_k_cardinality (AWA2 / ResNet-50)
  * -
    -
    - Text
    - qnli_top_k_cardinality (QNLI / BERT)
  * - `ModelRandomizationMetric <docs_api/quanda.metrics.heuristics.model_randomization.html>`_
    - Heuristic
    - Vision
    - mnist_model_randomization (MNIST / LeNet), cifar_model_randomization (CIFAR-10 / ResNet-9), awa2_model_randomization (AWA2 / ResNet-50)
  * -
    -
    - Text
    - qnli_model_randomization (QNLI / BERT)
  * - `MixedDatasetsMetric <docs_api/quanda.metrics.heuristics.mixed_datasets.html>`_
    - Heuristic
    - Vision
    - mnist_mixed_datasets (MNIST / LeNet), cifar_mixed_datasets (CIFAR-10 / ResNet-9), awa2_mixed_datasets (AWA2 / ResNet-50)
  * -
    -
    - Text
    - qnli_mixed_datasets (QNLI / BERT)
  * - `ClassDetectionMetric <docs_api/quanda.metrics.downstream_eval.class_detection.html>`_
    - Downstream Task Evaluator
    - Vision
    - mnist_class_detection (MNIST / LeNet), cifar_class_detection (CIFAR-10 / ResNet-9), awa2_class_detection (AWA2 / ResNet-50)
  * -
    -
    - Text
    - qnli_class_detection (QNLI / BERT)
  * - `SubclassDetectionMetric <docs_api/quanda.metrics.downstream_eval.subclass_detection.html>`_
    - Downstream Task Evaluator
    - Vision
    - mnist_subclass_detection (MNIST / LeNet), cifar_subclass_detection (CIFAR-10 / ResNet-9), awa2_subclass_detection (AWA2 / ResNet-50)
  * - `MislabelingDetectionMetric <docs_api/quanda.metrics.downstream_eval.mislabeling_detection.html>`_
    - Downstream Task Evaluator
    - Vision
    - mnist_mislabeling_detection (MNIST / LeNet), cifar_mislabeling_detection (CIFAR-10 / ResNet-9), awa2_mislabeling_detection (AWA2 / ResNet-50)
  * -
    -
    - Text
    - qnli_mislabeling_detection (QNLI / BERT)
  * - `ShortcutDetectionMetric <docs_api/quanda.metrics.downstream_eval.shortcut_detection.html>`_
    - Downstream Task Evaluator
    - Vision
    - mnist_shortcut_detection (MNIST / LeNet), cifar_shortcut_detection (CIFAR-10 / ResNet-9), awa2_shortcut_detection (AWA2 / ResNet-50)
  * - `MRRMetric <docs_api/quanda.metrics.downstream_eval.mrr.html>`_
    - Downstream Task Evaluator
    - Causal LM
    - gpt2_trex_openwebtext_ft_mrr (T-REx / GPT-2 fine-tuned on OpenWebText)
  * - `RecallAtKMetric <docs_api/quanda.metrics.downstream_eval.recall_at_k.html>`_
    - Downstream Task Evaluator
    - Causal LM
    - gpt2_trex_openwebtext_ft_recall_at_k (T-REx / GPT-2 fine-tuned on OpenWebText)
  * - `TailPatchMetric <docs_api/quanda.metrics.downstream_eval.tail_patch.html>`_
    - Downstream Task Evaluator
    - Causal LM
    - gpt2_trex_openwebtext_ft_tail_patch (T-REx / GPT-2 fine-tuned on OpenWebText)
  * - `LinearDatamodelingMetric <docs_api/quanda.metrics.ground_truth.linear_datamodeling.html>`_
    - Ground Truth
    - Vision
    - mnist_linear_datamodeling (MNIST / LeNet), cifar_linear_datamodeling (CIFAR-10 / ResNet-9), awa2_linear_datamodeling (AWA2 / ResNet-50)
  * -
    -
    - Text
    - qnli_linear_datamodeling (QNLI / BERT)


.. toctree::
   :caption: Usage
   :maxdepth: 2

   quickstart
   explainers
   tutorials

.. toctree::
   :caption: API Reference
   :maxdepth: 1

   docs_api/modules

.. toctree::
   :caption: Background

   background
   how_to_evaluate

.. toctree::
   :caption: Community

   contributing

