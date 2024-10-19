#!/usr/bin/env bash
#SBATCH --account=bbpa-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --mail-user=acbruce@ucsc.edu
#SBATCH --mail-type=ALL
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=4
#SBATCH --job-name=andy_openmm
#SBATCH --array=0-43

ANDY_IN_DIR="/projects/bbpa/acbruce/md_data/tica_sampled_starting_poses/wwdomain1_TYR_fixed"
ANDY_OUT_DIR="/projects/bbpa/acbruce/md_data/generate/wwdomain1"

export OMP_NUM_THREADS=32
echo "TASK_ID", $SLURM_ARRAY_TASK_ID

# Because we're mapping the GPUs to subprocesses the pool size needs to be a multiple of the number of GPUs
srun python3 batch_generate.py \
     ./andy.json \
     --batch-index $SLURM_ARRAY_TASK_ID \
     --batch-size 25 \
     --pool-size 8 \
     --gpus="0,1,2,3" \
     --remove-ligands \
     --prepare \
     --integrator integrator_4fs.json \
     --steps=1000000 \
     --report-steps=100 \
     --timeout 1.8 \
     --input-dir "${ANDY_IN_DIR}" \
     --data-dir "${ANDY_OUT_DIR}"
