#!/bin/bash
# Explainer sweep definitions for MNIST LeNet benchmark evaluation.
# Source this file, then use: ${EXPL_SWEEP[method]}
# Values are Hydra multirun overrides on explainer.kwargs.*

declare -A EXPL_SWEEP

EXPL_SWEEP[similarity]="explainer.kwargs.layers=fc_1,fc_2 explainer.kwargs.similarity_metric._target_=hydra.utils.get_method explainer.kwargs.similarity_metric.path=quanda.utils.functions.cosine_similarity,quanda.utils.functions.dot_product_similarity"
EXPL_SWEEP[representer_points]="explainer.kwargs.features_layer=relu_3,relu_4"
EXPL_SWEEP[tracincpfast]=""
EXPL_SWEEP[arnoldi]="explainer.kwargs.projection_dim=25,50,100 explainer.kwargs.arnoldi_dim=50,100,200"
EXPL_SWEEP[trak]="explainer.kwargs.proj_dim=512,1024,2048,4096"
EXPL_SWEEP[random]="explainer.kwargs.seed=0,1,42"
