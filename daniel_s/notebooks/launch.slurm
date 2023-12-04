#!/bin/bash
#SBATCH -A m4229
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 11:58:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=map_gpu:0,1,2,3
#SBATCH --job-name=daniel_openmm
#SBATCH --output=slurm.out
#SBATCH --array=0

eval "$(~/bin/micromamba shell hook -s posix)"
micromamba activate openmm

#export OMP_NUM_THREADS=2  # if code is not multithreaded, otherwise set to 8 or 16
export OMP_NUM_THREADS=32
echo "TASK_ID", $SLURM_ARRAY_TASK_ID
srun -N 1 -n 1 python3 generate_data.py $SLURM_ARRAY_TASK_ID
# py-torch example, --ntasks-per-node=1 --cpus-per-task=64
# srun python3 multiple_gpu.py

# nodes mean number of nodes NODES NODELIST(REASON) ex  4 gpub[020,030,035,038]
# -N means number of nodes to run on 
# -n means number of tasks to run

# -N is number of nodes
# -c 32 CPUs per task
