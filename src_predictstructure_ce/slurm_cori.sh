#!/bin/bash -l
#SBATCH -p regular
#SBATCH --qos=premium
#SBATCH -N 1
#SBATCH -A m2219
#SBATCH -t 10:00:00
#SBATCH -C haswell

# send mail for event 
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eunseok.lee@uah.edu

module load impi

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=spread
export OMP_PLACES=threads
export I_MPI_PMI_LIBRARY=/usr/lib64/slurmpmi/libpmi.so

srun -n 32 ./run_predictstructure_par

