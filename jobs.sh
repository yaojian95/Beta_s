#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=1:00:00
#SBATCH --nodes=4
#SBATCH --constraint=haswell

#SBATCH --account=mp107
#SBATCH --job-name=real_beta_s_SPASS_128

#SBATCH --array=14
#SBATCH -o /global/cscratch1/sd/jianyao/Data/sbatch/output_%A_%a.out

module load python 
source activate /global/cscratch1/sd/jianyao/my_test

cd /global/homes/j/jianyao/foreground

mpirun -n 248 python Test_exe_file_mpi.py --frelist spass_only --npix 4 --seed $SLURM_ARRAY_TASK_ID

