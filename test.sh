#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=PAS0061
#SBATCH --time=5:00
#SBATCH --mail-type=ALL
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128G

source activate trains
python osc_test.py