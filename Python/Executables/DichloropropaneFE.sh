#!/bin/bash
#SBATCH -C clk
#SBATCH -n 1 #(number of tasks)
#SBATCH --job-name=DichloropropaneFE_array_1
#SBATCH --output=./Logs/DichloropropaneFE_array_1_%A_%a.txt
#SBATCH -t 72:00:00 
#SBATCH -c 12  #(number of nodes)
#SBATCH --mem=144G
#SBATCH --array=0-199

# Variables en orden: ntrunc & num_beta / nshots / method / nlayers 

python3 ../Files/Simulation_DichloropropaneFE.py $SLURM_ARRAY_TASK_ID 2048 SLSQP 1 
