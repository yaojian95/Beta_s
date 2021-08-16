#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --constraint=haswell

#SBATCH --account=mp107
#SBATCH --job-name=beta_s_realizations

#SBATCH --array=10-49
#SBATCH -o /global/cscratch1/sd/jianyao/CBASS/sbatch_output/output_%A_%a.out

module load python 
source activate /global/homes/j/jianyao/my3.8

cd /global/homes/j/jianyao/foreground

mpirun -n 45 python Test_exe_file_mpi.py --frelist both --npix 39 --seed $SLURM_ARRAY_TASK_ID