#!/bin/bash
# Explainer sweep definitions for MNIST LeNet benchmark evaluation.
# Source this file, then use: ${EXPL_SWEEP[method]}
# Values are Hydra multirun overrides on explainer.kwargs.*

declare -A EXPL_SWEEP

EXPL_SWEEP[similarity]="explainer.kwargs.layers=fc_2,fc_3 device=cuda:1"
EXPL_SWEEP[representer_points]="explainer.kwargs.features_layer=relu_3,relu_4 device=cuda:1"
EXPL_SWEEP[tracincpfast]="device=cuda:0"
EXPL_SWEEP[arnoldi]="explainer.kwargs.projection_dim=50 explainer.kwargs.arnoldi_dim=100,200 device=cuda:0"
EXPL_SWEEP[trak]="explainer.kwargs.proj_dim=512,1024,2048 device=cuda:0"
EXPL_SWEEP[random]="device=cuda:1"
