Validity Checks
===============

Every benchmark exposes a ``sanity_check()`` method that verifies the released
assets behave as the benchmark assumes. The numbers below were measured on the
published assets.

What the checks mean
--------------------

.. list-table::
  :header-rows: 1

  * - Check
    - Meaning
  * - Train / validation accuracy
    - Accuracy of the released checkpoint on the benchmark's own train and
      validation splits.
  * - Mislabeling memorization
    - Accuracy on the mislabeled training samples with respect to their
      flipped labels. A high value means the model really memorized the noisy
      labels that the benchmark asks a TDA method to find.
  * - Shortcut memorization (train / eval)
    - Accuracy on the shortcut-patched training samples, and the share of
      shortcut-patched eval samples that the model predicts as the shortcut
      class. Both should be high, otherwise there is no shortcut behaviour to
      attribute.
  * - Adversarial memorization (train / eval)
    - The same quantities for the adversarial dataset in the mixed-datasets
      setting.
  * - Subset model accuracies
    - Validation accuracy of three of the 100 retrained LDS subset models, as
      a spot check that the counterfactual models trained properly.
  * - Eval post-filter
    - Share of the original eval split that survives the benchmark's eval
      filter. These benchmarks only evaluate on samples where the effect under
      study is actually present: subclass detection keeps the samples the
      model classifies correctly, and shortcut detection keeps the samples
      whose true class is not the shortcut class but which the model
      nonetheless predicts as the shortcut class. The percentage therefore
      says how much of the eval split the reported score is computed on -- a
      low value means a small effective eval set, not a broken benchmark.

Measured values
---------------

MNIST / LeNet
-------------

.. list-table::
  :header-rows: 1

  * - Benchmark
    - Train acc.
    - Val. acc.
    - Benchmark-specific checks
  * - TopKCardinality
    - 99.65%
    - 98.67%
    - —
  * - SubclassDetection
    - 99.78%
    - 99.18%
    - Eval post-filter: 99.17%
  * - MixedDatasets
    - 99.89%
    - 99.16%
    - Adversarial memorization (train / eval): 100.00% / 100.00%
  * - MislabelingDetection
    - 98.28%
    - 90.00%
    - Mislabeling memorization: 88.55%
  * - ShortcutDetection
    - 91.90%
    - 87.67%
    - Shortcut memorization (train / eval): 99.82% / 100.00%; Eval post-filter: 19.25%
  * - ClassDetection
    - 99.65%
    - 98.67%
    - —
  * - ModelRandomization
    - 99.65%
    - 98.67%
    - —
  * - LDS
    - 99.22%
    - 98.65%
    - Subset model accuracies: 98.25% / 97.85% / 98.34%
  * - LDS (α = 0.75)
    - 99.22%
    - 98.65%
    - Subset model accuracies: 98.61% / 98.46% / 98.45%
  * - LDS (α = 0.9)
    - 99.22%
    - 98.65%
    - Subset model accuracies: 98.08% / 98.81% / 98.58%
  * - LDS (α = 0.95)
    - 99.22%
    - 98.65%
    - Subset model accuracies: 98.48% / 97.83% / 98.64%
  * - LDS (α = 0.999)
    - 99.22%
    - 98.65%
    - Subset model accuracies: 98.75% / 98.41% / 98.51%

CIFAR-10 / ResNet-9
-------------------

.. list-table::
  :header-rows: 1

  * - Benchmark
    - Train acc.
    - Val. acc.
    - Benchmark-specific checks
  * - ClassDetection
    - 100.00%
    - 87.82%
    - —
  * - TopKCardinality
    - 100.00%
    - 87.82%
    - —
  * - ModelRandomization
    - 100.00%
    - 87.82%
    - —
  * - SubclassDetection
    - 100.00%
    - 93.60%
    - Eval post-filter: 93.46%
  * - MixedDatasets
    - 100.00%
    - 88.74%
    - Adversarial memorization (train / eval): 100.00% / 99.73%
  * - ShortcutDetection
    - 100.00%
    - 87.45%
    - Shortcut memorization (train / eval): 100.00% / 100.00%; Eval post-filter: 89.93%
  * - MislabelingDetection
    - 100.00%
    - 82.29%
    - Mislabeling memorization: 100.00%
  * - LDS
    - 100.00%
    - 87.82%
    - Subset model accuracies: 83.37% / 82.62% / 83.34%
  * - LDS (α = 0.75)
    - 100.00%
    - 87.82%
    - Subset model accuracies: 85.69% / 86.09% / 85.50%
  * - LDS (α = 0.9)
    - 100.00%
    - 87.82%
    - Subset model accuracies: 87.22% / 87.10% / 87.66%
  * - LDS (α = 0.95)
    - 100.00%
    - 87.82%
    - Subset model accuracies: 87.69% / 86.85% / 87.11%
  * - LDS (α = 0.999)
    - 100.00%
    - 87.82%
    - Subset model accuracies: 87.73% / 88.07% / 87.95%

AWA2 / ResNet-50
----------------

.. list-table::
  :header-rows: 1

  * - Benchmark
    - Train acc.
    - Val. acc.
    - Benchmark-specific checks
  * - ClassDetection
    - 99.91%
    - 80.23%
    - —
  * - TopKCardinality
    - 99.92%
    - 80.23%
    - —
  * - ModelRandomization
    - 99.93%
    - 80.23%
    - —
  * - SubclassDetection
    - 99.36%
    - 72.43%
    - Eval post-filter: 72.80%
  * - MixedDatasets
    - 99.91%
    - 79.78%
    - Adversarial memorization (train / eval): 100.00% / 94.65%
  * - ShortcutDetection
    - 99.91%
    - 79.37%
    - Shortcut memorization (train / eval): 100.00% / 100.00%; Eval post-filter: 93.94%
  * - MislabelingDetection
    - 99.91%
    - 77.38%
    - Mislabeling memorization: 99.93%
  * - LDS
    - 99.91%
    - 80.23%
    - Subset model accuracies: 69.32% / 69.77% / 68.60%

QNLI / BERT
-----------

.. list-table::
  :header-rows: 1

  * - Benchmark
    - Train acc.
    - Val. acc.
    - Benchmark-specific checks
  * - ClassDetection
    - 84.29%
    - 84.08%
    - —
  * - TopKCardinality
    - 84.29%
    - 84.08%
    - —
  * - ModelRandomization
    - 84.29%
    - 84.08%
    - —
  * - MislabelingDetection
    - 100.00%
    - 87.03%
    - Mislabeling memorization: 99.90%
  * - LDS
    - 84.29%
    - 84.08%
    - Subset model accuracies: 83.31% / 83.51% / 83.47%
  * - MixedDatasets
    - 85.07%
    - 84.47%
    - Adversarial memorization (train / eval): 99.35% / 98.91%

