#!/bin/bash
# Explainer sweep definitions for QNLI BERT benchmark evaluation.
# Source this file, then use: ${EXPL_SWEEP[method]}
# Values are Hydra multirun overrides on explainer.kwargs.*.
# BERT exposes `classifier` (final Linear) and `dropout` (pre-classifier
# pooled features); gradient-based methods are pinned to the classifier
# head to keep runtime tractable on a frozen BERT encoder.
# `task=text_classification` tells the wrappers to consume tokenized inputs.

declare -A EXPL_SWEEP

EXPL_SWEEP[similarity]="explainer.kwargs.layers=dropout,classifier +explainer.kwargs.task=text_classification device=cuda:0"
EXPL_SWEEP[representer_points]="explainer.kwargs.features_layer=dropout explainer.kwargs.classifier_layer=classifier +explainer.kwargs.task=text_classification device=cuda:0"
EXPL_SWEEP[tracincpfast]="+explainer.kwargs.task=text_classification device=cuda:0"
EXPL_SWEEP[arnoldi]="explainer.kwargs.layers=[classifier] explainer.kwargs.projection_dim=25,50 explainer.kwargs.arnoldi_dim=100,200 +explainer.kwargs.task=text_classification device=cuda:0"
EXPL_SWEEP[trak]="explainer.kwargs.proj_dim=512,1024,2048 +explainer.kwargs.task=text_classification device=cuda:0"
EXPL_SWEEP[random]="device=cuda:0"
