#!/bin/bash

#SBATCH --qos=regular
#SBATCH --time=9:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --constraint=haswell

#SBATCH --account=mp107

#SBATCH --job-name=beta_s


module load python 
source activate /global/homes/j/jianyao/myconda

cd /global/homes/j/jianyao/foreground

python exe_file-Copy1.py
