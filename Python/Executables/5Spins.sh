#!/bin/bash
#SBATCH -C clk
#SBATCH -n 1 #(number of tasks)
#SBATCH --job-name=5Spins_array_1
#SBATCH --output=./Logs/5Spins_array_1_%A_%a.txt
#SBATCH -t 150:10:00 
#SBATCH -c 12  #(number of nodes)
#SBATCH --mem-per-cpu=3G
#SBATCH --array=0-249

# Variables en orden: ntrunc & num_beta / number_of_spins / J / nshots / method / nlayers 

python3 ../Files/Simulation-CN.py $SLURM_ARRAY_TASK_ID 5 100 2048 SLSQP 1 
