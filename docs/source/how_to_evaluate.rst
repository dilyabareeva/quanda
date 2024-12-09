How to Assess the Quality of Attributions?
==========

Evaluation of interpretability approaches is a challenging task, as it is often difficult to define a ground truth for interpretability. Although there are various demonstrations of TDA’s potential for interpretability and practical applications, the critical question of how TDA methods should be effectively evaluated remains open. While methods based on estimating counterfactual retraining effects have a well-defined ground truth, this ground truth is computationally demanding and is not feasibly computable for large scale experiments. To address these shortcomings, several approaches have been proposed by the community, which can be categorized into three groups:

.. raw:: html

    <details><summary><b><big>Ground truth</big></b></summary>

As some of the methods are designed to approximate LOO effects, ground truth can often be computed for TDA evaluation. As explained above, this counterfactual ground truth approach requires retraining the model multiple times on different subsets of the training data, which is computationally expensive. Additionally, this ground truth is shown to be dominated by noise in practical deep learning settings, due to the inherent stochasticity of a typical training process (`Basu et al., 2021 <https://openreview.net/forum?id=xHKVVHGDOEk>`_; `Nguyen et al., 2023 <https://proceedings.neurips.cc/paper_files/paper/2023/hash/ca774047bc3b46cc81e53ead34cd5d5a-Abstract-Conference.html>`_). The most straightforward example of ground truth metrics is the Leave-one-out (LOO) metric (`Koh and Liang, 2017 <https://proceedings.mlr.press/v70/koh17a.html>`_). Linear Datamodeling Score (LDS) (`Park et al., 2023 <https://proceedings.mlr.press/v202/park23c.html>`_) is another example of a ground truth metric that measures the correlation between the (grouped) attribution scores and the actual output of models trained on a limited number subsets of the training set, which helps with the computational demand of the metric, but doesn't solve the problem in its totality.

.. raw:: html

    </details>

.. raw:: html

    <details><summary><b><big>Downstream Task Evaluators</big></b></summary>

To remedy the challenges associated with ground truth evaluation, the literature proposes to assess the utility of a TDA method within the context of an end-task. The most commonly used evaluation criteria is Mislabeling Detection (`Koh and Liang, 2017 <https://proceedings.mlr.press/v70/koh17a.html>`_; `Yeh et al., 2018 <https://proceedings.neurips.cc/paper/2018/hash/8a7129b8f3edd95b7d969dfc2c8e9d9d-Abstract.html>`_; `Pruthi et al., 2020 <https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html>`_) which compares different TDA methods in terms of their usefulness for detecting mislabeled samples after training the network on a dataset of which the labels are deliberately poisoned. Other examples could be detecting backdoor attacks (`Karthikeyan et al., 2021 <https://arxiv.org/abs/2111.04683>`_; `Yolcu et al., 2024 <https://arxiv.org/abs/2402.12118>`_) or predicting the model decision from its attributions (`Hanawa et al., 2021 <https://openreview.net/forum?id=9uvhpyQwzM_>`_).

.. raw:: html

    </details>

.. raw:: html

    <details><summary><b><big>Heuristics</big></b></summary>

Finally, the community also made use of heuristics (desirable properties or sanity checks) to evaluate the quality of TDA techniques. These include comparing the attributions of a trained model and a randomized model (`Hanawa et al., 2021 <https://openreview.net/forum?id=9uvhpyQwzM_>`_) and measuring the amount of overlap between the attributions for different test samples (`Barshan et al., 2020 <http://proceedings.mlr.press/v108/barshan20a/barshan20a.pdf>`_).

.. raw:: html

    </details>

|quanda| is designed to meet the need of a comprehensive and systematic evaluation framework, allowing practitioners and researchers to obtain a detailed view of the performance of TDA methods in various contexts.
