#!/usr/bin/env bash

set -e

IN_DIR="/media/DATA_18_TB_2/andy/tica_sampled_starting_poses/BBA_TYR_fixed/"
OUT_DIR="/media/DATA_18_TB_1/andy/benchmark_set_5/BBA"

rm -fr "${OUT_DIR}"

./batch_generate.py \
    ./andy.json \
    --gpus 1,2,3 \
    --prepare \
    --integrator integrator_4fs.json \
    --remove-ligands \
    --steps 10000000 \
    --report-steps 100 \
    --input-dir "${IN_DIR}" \
    --data-dir "${OUT_DIR}"
