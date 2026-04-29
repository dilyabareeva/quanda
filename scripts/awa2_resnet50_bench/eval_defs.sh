#!/bin/bash
# Explainer sweep definitions for AwA2 ResNet50 benchmark evaluation.

declare -A EXPL_SWEEP

EXPL_SWEEP[similarity]="explainer.kwargs.layers=flatten device=cuda:0"
EXPL_SWEEP[representer_points]="explainer.kwargs.features_layer=flatten explainer.kwargs.classifier_layer=fc device=cuda:0 explainer.kwargs.normalize=true,false hydra.launcher.n_jobs=1"
EXPL_SWEEP[tracincpfast]="device=cuda:0"
EXPL_SWEEP[arnoldi]="explainer.kwargs.layers=[fc] explainer.kwargs.projection_dim=50 explainer.kwargs.arnoldi_dim=100 device=cuda:0 hydra.launcher.n_jobs=1"
EXPL_SWEEP[trak]="explainer.kwargs.proj_dim=1024,2048 device=cuda:0 hydra.launcher.n_jobs=1"
EXPL_SWEEP[random]="device=cuda:0 explainer.kwargs.seed=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40 hydra.launcher.n_jobs=2"
