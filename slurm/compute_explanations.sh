#!/bin/bash
apptainer build --fakeroot --force ./quanda_pre_build.sif ./quanda/slurm/apptainer/quanda_pre_build.def
apptainer build --fakeroot --force ./quanda_build.sif ./quanda/slurm/apptainer/quanda_build.def
sbatch ./quanda/slurm/compute_explanations.sbatch
