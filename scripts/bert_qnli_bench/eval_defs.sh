#!/bin/bash
# Explainer sweep definitions for QNLI BERT benchmark evaluation.
# Source this file, then use: ${EXPL_SWEEP[method]}
# Values are Hydra multirun overrides on explainer.kwargs.*.
# BertClassifier exposes `classifier` (final Linear) and `dropout`
# (pre-classifier pooled features); gradient-based methods are pinned to
# the classifier head to keep runtime tractable on a frozen BERT encoder.
# `task=text_classification` tells the wrappers to consume tokenized inputs.
# Dattri methods take a `loss_func` builder — a callable `(model) -> fn`
# wired via `hydra.utils.get_method`; see `quanda.explainers.wrappers.
# dattri_losses` for the available builders.

declare -A EXPL_SWEEP

EXPL_SWEEP[similarity]="explainer.kwargs.layers=dropout,classifier +explainer.kwargs.task=text_classification device=cuda:0"
EXPL_SWEEP[trak]="explainer.kwargs.proj_dim=512 +explainer.kwargs.use_half_precision=true +explainer.kwargs.task=text_classification device=cuda:0 explainer.kwargs.batch_size=8"
EXPL_SWEEP[random]="device=cuda:0"
EXPL_SWEEP[dattri_arnoldi]="+explainer.kwargs.task=text_classification +explainer.kwargs.layer_name=[classifier.weight,classifier.bias] explainer.kwargs.proj_dim=25,50 explainer.kwargs.max_iter=100,200 device=cuda:1"
EXPL_SWEEP[dattri_ekfac]="+explainer.kwargs.task=text_classification +explainer.kwargs.module_name=classifier device=cuda:0"
EXPL_SWEEP[dattri_tracin]="+explainer.kwargs.task=text_classification +explainer.kwargs.layer_name=[classifier.weight,classifier.bias] device=cuda:1"
EXPL_SWEEP[dattri_graddot]="+explainer.kwargs.task=text_classification +explainer.kwargs.layer_name=[classifier.weight,classifier.bias] device=cuda:0"
EXPL_SWEEP[dattri_gradcos]="+explainer.kwargs.task=text_classification +explainer.kwargs.layer_name=[classifier.weight,classifier.bias] device=cuda:1"
