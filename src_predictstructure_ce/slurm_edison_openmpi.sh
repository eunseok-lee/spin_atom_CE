#!/bin/bash -l
#SBATCH -p regular
#SBATCH --qos=normal
#SBATCH -N 1
#SBATCH -A m2219
#SBATCH -t 20:00:00

# send mail for event 
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eunseok.lee@uah.edu

module load openmpi/2.0.3

export OMPI_MCA_pml=cm 
export OMPI_MCA_btl=self,vader,tcp 
export OMP_NUM_THREADS=6

srun -n 24 ./run_predictstructure_par

