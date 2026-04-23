#!/bin/bash
# Explainer sweep definitions for QNLI BERT benchmark evaluation.
# Source this file, then use: ${EXPL_SWEEP[method]}
# Values are Hydra multirun overrides on explainer.kwargs.*.
# BertClassifier exposes `classifier` (final Linear) and `dropout`
# (pre-classifier pooled features); gradient-based methods are pinned to
# the classifier head to keep runtime tractable on a frozen BERT encoder.
# `task=text_classification` tells the wrappers to consume tokenized inputs.
# Similarity uses `dropout` (post-pooler 768-dim features, no-op in eval);
# `classifier` would reduce to cosine on 2-dim logits and collapses to the
# model's predicted-class label on binary QNLI.
# TRAK: `proj_dim=2048` is the library default — 512 was too low-rank on
# BERT's ~110M params. `lambda_reg=1e-5` guards `finalize_features` from
# the numerical instability of the zero-reg pseudoinverse, which bites
# harder under `use_half_precision=true`.
# Kronfluence tracks `bert.pooler.dense` + `classifier` (the 768×768 feature
# layer plus the head) so per-query gradients actually depend on the query's
# semantic features. Classifier-only collapses to a near-query-independent
# hardness ranking on binary QNLI, so it is not swept.
# `batch_size=1000` (= max_eval_n) collapses the eval loop to a single
# `compute_pairwise_scores` call; otherwise each benchmark batch pays a
# full pass over the training set (~20 min × 125 batches).
# Random sweeps 20 seeds to characterize the null distribution of scores.
# RepresenterPoints extracts features from `dropout` and decomposes the
# final `classifier` linear head.

declare -A EXPL_SWEEP

EXPL_SWEEP[similarity]="explainer.kwargs.layers=dropout +explainer.kwargs.task=text_classification device=cuda:1"
EXPL_SWEEP[trak]="explainer.kwargs.proj_dim=512,2048 +explainer.kwargs.task=text_classification device=cuda:0 explainer.kwargs.batch_size=8"
EXPL_SWEEP[random]="device=cuda:1 explainer.kwargs.seed=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19"
EXPL_SWEEP[kronfluence]="explainer.kwargs.task_module.tracked_modules=[bert.pooler.dense,classifier] +explainer.kwargs.task=text_classification device=cuda:1 batch_size=1000"
EXPL_SWEEP[representer_points]="explainer.kwargs.features_layer=dropout explainer.kwargs.classifier_layer=classifier +explainer.kwargs.task=text_classification explainer.kwargs.batch_size=32 device=cuda:1"
