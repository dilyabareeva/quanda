#!/bin/bash

declare -A EXPL_SWEEP

EXPL_SWEEP[similarity]="explainer.kwargs.layers=dropout explainer.kwargs.similarity_metric.path=quanda.utils.functions.cosine_similarity,quanda.utils.functions.dot_product_similarity +explainer.kwargs.task=text_classification device=cuda:0 batch_size=32"
EXPL_SWEEP[trak]="explainer.kwargs.proj_dim=2048 explainer.kwargs.lambda_reg=1e-5 +explainer.kwargs.task=text_classification device=cuda:0 explainer.kwargs.batch_size=8"
EXPL_SWEEP[random]="device=cuda:0 explainer.kwargs.seed=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19"
EXPL_SWEEP[kronfluence]="explainer.kwargs.task_module.tracked_modules=[bert.pooler.dense,classifier] +explainer.kwargs.score_args._target_=kronfluence.arguments.ScoreArguments +explainer.kwargs.score_args.use_measurement_for_self_influence=true +explainer.kwargs.task=text_classification device=cuda:0 batch_size=1000"
EXPL_SWEEP[representer_points]="explainer.kwargs.features_layer=dropout explainer.kwargs.classifier_layer=classifier explainer.kwargs.normalize=true +explainer.kwargs.task=text_classification explainer.kwargs.batch_size=32 device=cuda:0"
