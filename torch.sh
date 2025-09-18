#!/bin/bash
#SBATCH --job-name=one_comparison
#SBATCH --account=PAS0061
#SBATCH --time=3:00:00
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --gpus-per-node=4

source activate trains
python script.py