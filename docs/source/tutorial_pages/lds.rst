Linear Datamodeling Score (LDS)
===============================

The :doc:`LinearDatamodelingMetric <../docs_api/quanda.metrics.ground_truth.linear_datamodeling>` is a ground-truth metric (Park et al., 2023): it measures how well an attribution method predicts the actual change in model output when retraining on different subsets of the training data. The corresponding :doc:`LinearDatamodeling <../docs_api/quanda.benchmarks.ground_truth.linear_datamodeling>` benchmark wraps this end-to-end — including training the counterfactual subset models — and is the most expensive benchmark in |quanda|. This page collects the practical caveats you should know before running it.

Caveats
-------

**1. ``load_pretrained`` downloads M subset checkpoints.** LDS retrains the model on ``M`` random subsets of the training data (default ``m=100`` in the published configs). ``LinearDatamodeling.load_pretrained(...)`` therefore pulls down ``M+1`` checkpoints from the Hugging Face Hub (the main model plus every subset). For the ``mnist_linear_datamodeling`` / ``cifar_linear_datamodeling`` / ``awa2_linear_datamodeling`` / ``qnli_linear_datamodeling`` benchmarks this is 100 subset checkpoints; expect a sizeable download and disk footprint on first use.

**2. Counterfactual subset logits should be precomputed once and reused.** During evaluation, each subset model is run over the eval dataset to produce *counterfactual logits*, which are then correlated with the explainer's group attributions. These per-subset logits depend only on the subset checkpoints and the eval subsample — they do **not** depend on the explainer being evaluated. Recomputing them inside every ``evaluate(...)`` call is wasteful, so :class:`LinearDatamodeling` exposes ``cache_subset_logits`` that runs the M forward passes once and writes them to disk; subsequent ``evaluate(...)`` calls pass that directory via ``subset_logits_dir=...`` and skip the recomputation.

For embarrassingly parallel pipelines, ``cache_subset_logits_per_idx`` does the same but for a single subset index, so the M forward passes can be sharded across workers.

**3. Training subset models from scratch.** Calling ``LinearDatamodeling.train(config)`` trains the main model and then iterates through all ``M`` subset models sequentially in the same process — fine for small benchmarks but typically the wrong shape for production runs.

The recommended pattern is to split training into two phases. ``train(config, skip_subsets=True)`` trains and persists the main model and writes the metadata (split ids, ``subset_ids.yaml``) but does **not** train any subset models. Then ``train_subset(config, idx=...)`` rebuilds the benchmark from that metadata, trains a single subset ``idx``, and persists its checkpoint. ``train_subset`` is designed to be the unit of work for an array job — call it once per worker / GPU / SLURM task and the M subsets train in parallel rather than back-to-back. Remember that the passed config should contain a ``bench_save_dir`` field that is used to save the main model, the subset checkpoints, and the metadata that links them together.

.. literalinclude:: ../../../tests/integration/test_benchmark_integration.py
   :language: python
   :start-after: # START19
   :end-before: # END19
   :dedent:

.. literalinclude:: ../../../tests/integration/test_benchmark_integration.py
   :language: python
   :start-after: # START20
   :end-before: # END20
   :dedent:

The companion ``scripts/train_lds_subset.py`` wraps ``train_subset`` as a CLI entry point that takes a single ``--idx``, intended for SLURM array jobs.

Both methods accept either a config dict, a registered ``bench_id``, or a path to a benchmark YAML.

Precomputing and reusing subset logits
--------------------------------------

The example below loads the published ``mnist_linear_datamodeling`` benchmark via ``load_pretrained``, then calls ``cache_subset_logits`` and ``explain`` directly on the loaded benchmark to populate both caches. The subsequent ``evaluate`` call reads from both. The same ``subset_logits_dir`` can be passed to every ``evaluate(...)`` call regardless of explainer; the same explanations ``cache_dir`` + ``use_cached_expl=True`` can be reused whenever the explainer / ``expl_kwargs`` / eval-subsample match.

.. literalinclude:: ../../../tests/integration/test_benchmark_integration.py
   :language: python
   :start-after: # START17
   :end-before: # END17
   :dedent:

.. literalinclude:: ../../../tests/integration/test_benchmark_integration.py
   :language: python
   :start-after: # START18
   :end-before: # END18
   :dedent:

When several explainers will be benchmarked against the same LDS setup, call ``cache_subset_logits`` once and feed the returned directory into every ``evaluate(...)``. Likewise, call ``explain`` once per explainer and reuse the returned cache directory across re-evaluations or across sibling benchmarks that share the same model + train/eval datasets via a common ``explanations_group`` in the YAML.

All of the classmethods on this page (``load_pretrained``, ``train``, ``train_subset``, ``explain``, ``cache_subset_logits``, ``cache_subset_logits_per_idx``) accept ``bench_id`` / ``config`` either as a registered string (e.g. ``"mnist_linear_datamodeling"``), a path to a benchmark YAML, or a config dict.
