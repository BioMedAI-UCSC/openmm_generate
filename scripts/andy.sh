#!/usr/bin/env bash

#mdconvert -o /media/DATA_18_TB_2/andy/benchmark_generate_input_2/chignolin.pdb --topology /media/DATA_18_TB_2/andy/benchmark_set_2/trajectory_datas/chignolin/extract/filtered/filtered.psf /media/DATA_18_TB_2/andy/benchmark_set_2/trajectory_datas/chignolin/extract/filtered/filtered.pdb

rm -r ../data
./batch_generate.py \
    ./andy.json \
    --prepare \
    --remove-ligands \
    --steps 1000000 \
    --input-dir /media/DATA_18_TB_2/andy/benchmark_generate_input_3
