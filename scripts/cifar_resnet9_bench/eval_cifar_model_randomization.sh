#!/bin/bash
# Model randomization sanity check on CIFAR-10 ResNet9.


N_RAND_MODELS=10

# Cheapest first, so results land early and the two long ones run last.
methods=(
    similarity
    representer_points
    trak
    tracincpfast
    arnoldi
)

source "$(dirname "$0")/eval_defs.sh"

EVAL_CONFIG_NAME="cifar_resnet9"

benchmarks=(
    cifar_model_randomization
)

methods=("$METHOD")
PARALLEL=false

# Pin the previously-selected best hyperparameters instead of sweeping.
EXPL_SWEEP[representer_points]="${EXPL_SWEEP[representer_points]//normalize=true,false/normalize=false}"
EXPL_SWEEP[arnoldi]="${EXPL_SWEEP[arnoldi]//"layers=[model.9],[model.6,model.9]"/layers=[model.9]}"
EXPL_SWEEP[arnoldi]="${EXPL_SWEEP[arnoldi]//projection_dim=50,100/projection_dim=100}"
EXPL_SWEEP[trak]="${EXPL_SWEEP[trak]//proj_dim=1024,2048/proj_dim=2048}"

EXPL_SWEEP[$METHOD]="${EXPL_SWEEP[$METHOD]//cuda:1/cuda:0}"
EXPL_SWEEP[$METHOD]="${EXPL_SWEEP[$METHOD]//hydra.launcher.n_jobs=4/hydra.launcher.n_jobs=1}"
EXPL_SWEEP[$METHOD]="${EXPL_SWEEP[$METHOD]} results_dir=\${root_dir}/eval_results/cifar_mrand_n${N_RAND_MODELS}"
EXPL_SWEEP[$METHOD]="${EXPL_SWEEP[$METHOD]} +n_rand_models=${N_RAND_MODELS} +bootstrap=false"

source "$(dirname "$0")/../eval.sh"
