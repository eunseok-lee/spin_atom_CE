#!/bin/bash 
# 
##SBATCH -p general # partition (queue) 
#SBATCH -N 1 # number of nodes 
#SBATCH -n 24 # number of cores 
#SBATCH --mem 64000 # memory pool for all cores 
#SBATCH -t 1-00:00 # time (D-HH:MM) 
#SBATCH -o job.%N.%j.out # STDOUT 
#SBATCH -e job.%N.%j.err # STDERR

# send mail for event 
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=eunseok.lee@uah.edu

module load openmpi/gcc/64/1.8.5

mpirun -np 24 run_predictstructure_par
