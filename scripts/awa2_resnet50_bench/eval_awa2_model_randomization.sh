#!/bin/bash
# Model randomization sanity check on AwA2 ResNet50.

N_RAND_MODELS=10

methods=(
    similarity
    representer_points
    tracincpfast
    arnoldi
    trak
    kronfluence
)

source "$(dirname "$0")/eval_defs.sh"

EVAL_CONFIG_NAME="awa2_resnet50"

benchmarks=(
    awa2_model_randomization
)

methods=("$METHOD")
PARALLEL=false

EXPL_SWEEP[representer_points]="${EXPL_SWEEP[representer_points]//normalize=true,false/normalize=true}"
EXPL_SWEEP[trak]="${EXPL_SWEEP[trak]//proj_dim=1024,2048/proj_dim=4096}"

DEVICE="${DEVICE:-cuda:0}"
EXPL_SWEEP[$METHOD]="${EXPL_SWEEP[$METHOD]//device=cuda:1/device=$DEVICE}"
EXPL_SWEEP[$METHOD]="${EXPL_SWEEP[$METHOD]//device=cuda:0/device=$DEVICE}"


EXPL_SWEEP[$METHOD]="${EXPL_SWEEP[$METHOD]} results_dir=\${root_dir}/eval_results/awa2_mrand_n${N_RAND_MODELS}"
EXPL_SWEEP[$METHOD]="${EXPL_SWEEP[$METHOD]} +n_rand_models=${N_RAND_MODELS} +bootstrap=false"

source "$(dirname "$0")/../eval.sh" "$@"
