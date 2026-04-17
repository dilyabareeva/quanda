#!/bin/bash
# Explainer sweep definitions for CIFAR-10 ResNet9 benchmark evaluation.
# Source this file, then use: ${EXPL_SWEEP[method]}
# Values are Hydra multirun overrides on explainer.kwargs.*.
# Layer names follow ResNet9's Sequential submodule indices
# (model.8 = Flatten output, model.9 = final Linear).

declare -A EXPL_SWEEP

EXPL_SWEEP[similarity]="explainer.kwargs.layers=model.8,model.9 device=cuda:1"
EXPL_SWEEP[representer_points]="explainer.kwargs.features_layer=model.8 explainer.kwargs.classifier_layer=model.9 device=cuda:1"
EXPL_SWEEP[tracincpfast]="device=cuda:0"
EXPL_SWEEP[arnoldi]="explainer.kwargs.layers=[model.9] explainer.kwargs.projection_dim=50 explainer.kwargs.arnoldi_dim=100,200 device=cuda:0"
EXPL_SWEEP[trak]="explainer.kwargs.proj_dim=512,1024,2048 device=cuda:0"
EXPL_SWEEP[random]="device=cuda:1"
