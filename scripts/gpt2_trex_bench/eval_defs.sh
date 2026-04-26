#!/bin/bash
# Per-explainer Hydra sweeps for the GPT-2 / TREx fact-tracing benchmarks.

declare -A EXPL_SWEEP

LOSS_PATH="quanda.explainers.wrappers.dattri_losses"

# Common dattri overrides — apply to every dattri-family explainer that
# uses one of the BERT base yamls.
DATTRI_BASE='+explainer.kwargs.task=causal_lm +explainer.kwargs.hf_input_keys=[input_ids,attention_mask] explainer.kwargs.batch_size=1'

EXPL_SWEEP[random]="explainer.kwargs.seed=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 device=cuda:0"

EXPL_SWEEP[similarity]="explainer.kwargs.layers=transformer.ln_f explainer.kwargs.similarity_metric.path=quanda.utils.functions.cosine_similarity,quanda.utils.functions.dot_product_similarity +explainer.kwargs.task=causal_lm device=cuda:0 batch_size=32"

EXPL_SWEEP[dattri_arnoldi]="explainer.kwargs.loss_func.path=${LOSS_PATH}.causal_lm_batched_loss explainer.kwargs.precompute_data_ratio=1.0 ${DATTRI_BASE} device=cuda:0"

EXPL_SWEEP[dattri_tracin]="explainer.kwargs.loss_func.path=${LOSS_PATH}.causal_lm_per_sample_loss explainer.kwargs.learning_rate=1.0e-05 ${DATTRI_BASE} device=cuda:0"

EXPL_SWEEP[dattri_graddot]="explainer.kwargs.loss_func.path=${LOSS_PATH}.causal_lm_per_sample_loss ${DATTRI_BASE} device=cuda:0"

EXPL_SWEEP[dattri_gradcos]="explainer.kwargs.loss_func.path=${LOSS_PATH}.causal_lm_per_sample_loss ${DATTRI_BASE} device=cuda:0"

# Methods with dedicated GPT-2 yamls (defaults already correct):
EXPL_SWEEP[kronfluence_gpt2]="device=cuda:0 batch_size=1"
EXPL_SWEEP[dattri_trak]="device=cuda:0 +explainer.kwargs.layer_name=['transformer.h.11.mlp.c_fc.weight','transformer.h.11.mlp.c_proj.weight'] explainer.kwargs.projector_kwargs.proj_dim=1024"
EXPL_SWEEP[dattri_if_explicit]="device=cuda:0"
EXPL_SWEEP[dattri_if_cg]="device=cuda:0 batch_size=32"
EXPL_SWEEP[dattri_if_lissa]="device=cuda:0"
EXPL_SWEEP[dattri_if_datainf]="device=cuda:0 +explainer.kwargs.layer_name=['transformer.h.11.mlp.c_fc.weight','transformer.h.11.mlp.c_proj.weight']"
