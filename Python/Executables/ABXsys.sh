#!/bin/bash
#SBATCH -C clk
#SBATCH -n 1 #(number of tasks)
#SBATCH --job-name=ABXsys_array_1
#SBATCH --output=./Logs/ABXsys_array_1_%A.txt
#SBATCH -t 00:05:00 
#SBATCH -c 3  #(number of nodes)
#SBATCH --mem-per-cpu=500M
#SBATCH --array=0-249

# Variables en orden: ntrunc & num_beta / system_name / nshots / method / nlayers 

python3 ../Files/Simulation.py $SLURM_ARRAY_TASK_ID ABX 0 SLSQP 1 
