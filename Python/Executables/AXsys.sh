#!/bin/bash
#SBATCH -C clk
#SBATCH -n 1 #(number of tasks)
#SBATCH --job-name=AXsys_array_1
#SBATCH --output=./Logs/AXsys_array_1_%A.txt
#SBATCH -t 00:03:00 
#SBATCH -c 3  #(number of nodes)
#SBATCH --mem-per-cpu=500M
#SBATCH --array=0-249

# Variables en orden: ntrunc & num_beta / system_name / nshots / method / nlayers 

python3 ../Files/Simulation.py $SLURM_ARRAY_TASK_ID AX 0 SLSQP 1 
