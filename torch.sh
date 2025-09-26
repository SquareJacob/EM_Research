#!/bin/bash
#SBATCH --job-name=one_comparison
#SBATCH --account=PAS0061
#SBATCH --time=3:00
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --mem=256G
#SBATCH --gpus-per-node=4

source activate trains
torchrun --nproc_per_node=1 script.py