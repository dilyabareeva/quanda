#!/bin/bash
# Explainer sweep definitions for AwA2 ResNet50 benchmark evaluation.

declare -A EXPL_SWEEP

EXPL_SWEEP[similarity]="explainer.kwargs.layers=flatten explainer.kwargs.batch_size=128 device=cuda:0 hydra.launcher.n_jobs=1 batch_size=128"
EXPL_SWEEP[representer_points]="explainer.kwargs.features_layer=flatten explainer.kwargs.classifier_layer=fc explainer.kwargs.batch_size=128 device=cuda:1 explainer.kwargs.normalize=true,false hydra.launcher.n_jobs=1 batch_size=128 +explainer.kwargs.random_init=true"
EXPL_SWEEP[tracincpfast]="explainer.kwargs.batch_size=256 batch_size=256 device=cuda:0"
EXPL_SWEEP[arnoldi]="explainer.kwargs.layers=[fc] explainer.kwargs.projection_dim=50 explainer.kwargs.arnoldi_dim=100 explainer.kwargs.batch_size=256 +explainer.kwargs.precompute_data_ratio=0.1 device=cuda:1 hydra.launcher.n_jobs=1"
EXPL_SWEEP[trak]="explainer.kwargs.proj_dim=1024,2048,4096 explainer.kwargs.batch_size=32 device=cuda:0 hydra.launcher.n_jobs=1"
EXPL_SWEEP[random]="device=cuda:0 explainer.kwargs.seed=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40 hydra.launcher.n_jobs=5 batch_size=128 device=cuda:1"
EXPL_SWEEP[kronfluence]="explainer.kwargs.task_module._target_=quanda.explainers.wrappers.kronfluence_tasks.ImageClassificationTask explainer.kwargs.task_module.tracked_modules=[layer4.2.conv3,fc] +explainer.kwargs.score_args._target_=kronfluence.arguments.ScoreArguments +explainer.kwargs.score_args.use_measurement_for_self_influence=true explainer.kwargs.batch_size=64 device=cuda:0 batch_size=1000 inference_batch_size=64 hydra.launcher.n_jobs=1"
