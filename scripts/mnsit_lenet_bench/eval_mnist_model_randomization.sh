#!/bin/bash
# Model randomization sanity check on MNIST LeNet.

N_RAND_MODELS=10


methods=(
    similarity
    representer_points
    trak
    tracincpfast
    arnoldi
)


source "$(dirname "$0")/eval_defs.sh"

EVAL_CONFIG_NAME="mnist_lenet"

benchmarks=(
    mnist_model_randomization
)

methods=("$METHOD")
PARALLEL=false

# Pin the previously-selected best hyperparameters instead of sweeping.
EXPL_SWEEP[similarity]="${EXPL_SWEEP[similarity]//layers=relu_3,fc_2,relu_4/layers=relu_4}"
EXPL_SWEEP[representer_points]="${EXPL_SWEEP[representer_points]//features_layer=fc_2,relu_4/features_layer=relu_4}"
EXPL_SWEEP[representer_points]="${EXPL_SWEEP[representer_points]//normalize=true,false/normalize=true}"
EXPL_SWEEP[arnoldi]="${EXPL_SWEEP[arnoldi]//"layers=[fc_3],[fc_1,fc_2,fc_3]"/layers=[fc_3]}"
EXPL_SWEEP[arnoldi]="${EXPL_SWEEP[arnoldi]//projection_dim=50,100/projection_dim=100}"
EXPL_SWEEP[trak]="${EXPL_SWEEP[trak]//proj_dim=512,1024/proj_dim=1024}"


EXPL_SWEEP[$METHOD]="${EXPL_SWEEP[$METHOD]//cuda:1/cuda:0}"
EXPL_SWEEP[$METHOD]="${EXPL_SWEEP[$METHOD]//hydra.launcher.n_jobs=4/hydra.launcher.n_jobs=1}"
EXPL_SWEEP[$METHOD]="${EXPL_SWEEP[$METHOD]} results_dir=\${root_dir}/eval_results/mnist_mrand_n${N_RAND_MODELS}"
EXPL_SWEEP[$METHOD]="${EXPL_SWEEP[$METHOD]} +n_rand_models=${N_RAND_MODELS} +bootstrap=false"

source "$(dirname "$0")/../eval.sh"
