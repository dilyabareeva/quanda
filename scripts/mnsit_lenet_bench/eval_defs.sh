#!/bin/bash
# Explainer sweep definitions for MNIST LeNet benchmark evaluation.
# Source this file, then use: ${EXPL_SWEEP[method]}
# Values are Hydra multirun overrides on explainer.kwargs.*

declare -A EXPL_SWEEP

EXPL_SWEEP[similarity]="explainer.kwargs.layers=fc_1,relu_3,fc_2,relu_4,fc_3 device=cuda:0"
EXPL_SWEEP[representer_points]="explainer.kwargs.features_layer=fc_2,relu_4 explainer.kwargs.normalize=true,false device=cuda:0"
EXPL_SWEEP[tracincpfast]="device=cuda:0"
EXPL_SWEEP[arnoldi]="explainer.kwargs.projection_dim=50,100 explainer.kwargs.arnoldi_dim=200 device=cuda:0 explainer.kwargs.layers=[fc_3],[fc_1,fc_2,fc_3]"
EXPL_SWEEP[trak]="explainer.kwargs.proj_dim=512,1024 device=cuda:0"
EXPL_SWEEP[random]="device=cuda:0 explainer.kwargs.seed=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19"
