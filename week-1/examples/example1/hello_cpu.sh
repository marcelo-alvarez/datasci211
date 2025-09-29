#!/bin/bash

# ---- Preprocessing Directives ---
# Anything prepended with the # character is a comment
# with the exception of lines starting with #SBATCH 
# which is a pre-processing directive the batch scheduler
# (Slurm) that Marlowe uses

#SBATCH -J hello-cpu        # Job name
#SBATCH -o hello-cpu.%j.out # Output file
#SBATCH -t 0-00:02:00       # Time limit format: DD-HH:MM:SS
#SBATCH -N 1                # Number of Nodes
#SBATCH --ntasks=2          # Number of tasks
#SBATCH --cpus-per-task=1   # Number of CPUs per Task
#SBATCH -A marlowe-c001     # Marlowe Slurm Account for DataSci 211
#SBATCH --partition=class   # Partition to be used for DataSci 211

# --- Load environment modules ---
module load conda

# --- Run commands --- 
echo "Job $SLURM_JOB_ID on $(date)"
echo "Node list: $SLURM_NODELIST"
echo "SLURM_NTASKS=$SLURM_NTASKS  SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo -n "CPU model: "; grep -m1 "model name" /proc/cpuinfo | sed 's/.*: //'

# Note the srun command here which will automatically run 
# whatever comes after it using the resources (cores) specified above
srun --label python3 pi_cpu.py

