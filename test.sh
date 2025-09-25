#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=PAS0061
#SBATCH --time=5:00
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4

source activate trains
torchrun --nproc_per_node=4 osc_test.py