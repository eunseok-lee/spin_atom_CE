#!/bin/bash -l
#SBATCH -p regular
#SBATCH --qos=normal
#SBATCH -N 1
#SBATCH -A m2219
#SBATCH -t 20:00:00

# send mail for event 
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eunseok.lee@uah.edu

module load impi

export I_MPI_FABRICS=ofi
export I_MPI_OFI_PROVIDER=gni
export I_MPI_OFI_LIBRARY=/usr/common/software/libfabric/1.5.0/gnu/lib/libfabric.so
export I_MPI_PMI_LIBRARY=/usr/lib64/slurmpmi/libpmi.so

export OMP_NUM_THREADS=6
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

srun -n 24 ./run_predictstructure_par

