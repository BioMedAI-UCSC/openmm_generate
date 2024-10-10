#!/usr/bin/env bash

#srun --account=bbpa-delta-gpu --partition=gpuA100x4 --gpus=4 --cpus-per-task=4 --mem=8g --time=12:00:00 ./andy.sh

set -e

IN_DIR="/u/acbruce/md_data/tica_sampled_starting_poses/chignolin_TYR_fixed"
OUT_DIR="/u/acbruce/md_data/generate/chignolin"

rm -fr "${OUT_DIR}"

./batch_generate.py \
    ./andy.json \
    --gpus 0,1,2,3 \
    --prepare \
    --integrator integrator_4fs.json \
    --remove-ligands \
    --steps 10000000 \
    --report-steps 100 \
    --batch-size 60 \
    --batch-index 0 \
    --input-dir "${IN_DIR}" \
    --data-dir "${OUT_DIR}"
