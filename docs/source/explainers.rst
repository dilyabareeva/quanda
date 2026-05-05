Explainer Wrappers
==================

|quanda| ships wrappers around several existing TDA libraries, exposing them
through a single :doc:`Explainer <docs_api/quanda.explainers.base>` interface.
The tables below list every wrapper class and cite the paper that introduced
the underlying method. All wrapper sources live under
`quanda/explainers/wrappers/
<https://github.com/dilyabareeva/quanda/tree/main/quanda/explainers/wrappers>`_.

All wrappers can be imported directly from ``quanda.explainers.wrappers``,
for example:

.. code:: python

   from quanda.explainers.wrappers import (
       CaptumSimilarity,
       TRAK,
       Kronfluence,
       RepresenterPoints,
       DattriIFExplicit,
   )

Captum
------
Wrappers around the influence methods provided by `Captum
<https://github.com/pytorch/captum/tree/master>`_. Source:
`captum_influence.py
<https://github.com/dilyabareeva/quanda/blob/main/quanda/explainers/wrappers/captum_influence.py>`_.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Wrapper
     - Reference
   * - ``CaptumSimilarity``
     - Similarity between test and training samples in the representation
       space of a chosen layer. See `Captum's SimilarityInfluence docs
       <https://captum.ai/api/influence.html#similarityinfluence>`__.
   * - ``CaptumArnoldi``
     - Schioppa et al., 2022. *Scaling Up Influence Functions.*
       `arXiv:2112.03052 <https://arxiv.org/abs/2112.03052>`__
   * - ``CaptumTracInCP``
     - Pruthi et al., 2020. *Estimating Training Data Influence by Tracing
       Gradient Descent.* `NeurIPS 2020
       <https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html>`__
   * - ``CaptumTracInCPFast``
     - Pruthi et al., 2020. *Estimating Training Data Influence by Tracing
       Gradient Descent.* `NeurIPS 2020
       <https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html>`__
   * - ``CaptumTracInCPFastRandProj``
     - Pruthi et al., 2020. *Estimating Training Data Influence by Tracing
       Gradient Descent.* `NeurIPS 2020
       <https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html>`__

Representer Point Selection
---------------------------
Source: `representer_points.py
<https://github.com/dilyabareeva/quanda/blob/main/quanda/explainers/wrappers/representer_points.py>`_.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Wrapper
     - Reference
   * - ``RepresenterPoints``
     - Yeh et al., 2018. *Representer Point Selection for Explaining Deep
       Neural Networks.* `NeurIPS 2018
       <https://proceedings.neurips.cc/paper/2018/hash/8a7129b8f3edd95b7d969dfc2c8e9d9d-Abstract.html>`__
       — original implementation: `chihkuanyeh/Representer_Point_Selection
       <https://github.com/chihkuanyeh/Representer_Point_Selection>`__

TRAK
----
Source: `trak_wrapper.py
<https://github.com/dilyabareeva/quanda/blob/main/quanda/explainers/wrappers/trak_wrapper.py>`_.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Wrapper
     - Reference
   * - ``TRAK``
     - Park et al., 2023. *TRAK: Attributing Model Behavior at Scale.*
       `ICML 2023 <https://proceedings.mlr.press/v202/park23c.html>`__
       — original implementation: `MadryLab/trak
       <https://github.com/MadryLab/trak>`__

Kronfluence
-----------
Source: `kronfluence.py
<https://github.com/dilyabareeva/quanda/blob/main/quanda/explainers/wrappers/kronfluence.py>`_.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Wrapper
     - Reference
   * - ``Kronfluence (incl. EK-FAC)``
     - Grosse et al., 2023. *Studying Large Language Model Generalization with
       Influence Functions.* `arXiv:2308.03296
       <https://arxiv.org/abs/2308.03296>`__
       — original implementation: `pomonam/kronfluence
       <https://github.com/pomonam/kronfluence>`__

Dattri
------
Wrappers around the unified TDA family provided by `Dattri
<https://github.com/TRAIS-Lab/dattri>`_ (Deng et al., 2024,
`arXiv:2410.04555 <https://arxiv.org/abs/2410.04555>`__). Source:
`dattri_influence.py
<https://github.com/dilyabareeva/quanda/blob/main/quanda/explainers/wrappers/dattri_influence.py>`_.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Wrapper
     - Reference
   * - ``DattriIFExplicit``
     - Koh and Liang, 2017. *Understanding Black-box Predictions via Influence
       Functions.* `ICML 2017
       <https://proceedings.mlr.press/v70/koh17a.html>`__
   * - ``DattriIFCG``
     - Koh and Liang, 2017 (conjugate-gradient solver). `ICML 2017
       <https://proceedings.mlr.press/v70/koh17a.html>`__
   * - ``DattriIFLiSSA``
     - Agarwal et al., 2017. *Second-Order Stochastic Optimization for Machine
       Learning in Linear Time.* `JMLR 2017
       <https://www.jmlr.org/papers/v18/16-491.html>`__
   * - ``DattriIFDataInf``
     - Kwon et al., 2024. *DataInf: Efficiently Estimating Data Influence in
       LoRA-tuned LLMs and Diffusion Models.* `ICLR 2024
       <https://openreview.net/forum?id=9m02ib92Wz>`__
   * - ``DattriArnoldi``
     - Schioppa et al., 2022. *Scaling Up Influence Functions.*
       `arXiv:2112.03052 <https://arxiv.org/abs/2112.03052>`__
   * - ``DattriEKFAC``
     - Grosse et al., 2023. *Studying Large Language Model Generalization with
       Influence Functions.* `arXiv:2308.03296
       <https://arxiv.org/abs/2308.03296>`__
   * - ``DattriTracInCP``
     - Pruthi et al., 2020. *Estimating Training Data Influence by Tracing
       Gradient Descent.* `NeurIPS 2020
       <https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html>`__
   * - ``DattriGradDot``
     - Charpiat et al., 2019. *Input Similarity from the Neural Network
       Perspective.* `NeurIPS 2019
       <https://proceedings.neurips.cc/paper_files/paper/2019/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html>`__
   * - ``DattriGradCos``
     - Charpiat et al., 2019. *Input Similarity from the Neural Network
       Perspective.* `NeurIPS 2019
       <https://proceedings.neurips.cc/paper_files/paper/2019/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html>`__
   * - ``DattriTRAK``
     - Park et al., 2023. *TRAK: Attributing Model Behavior at Scale.*
       `ICML 2023 <https://proceedings.mlr.press/v202/park23c.html>`__
