#!/bin/bash
if [ $# -lt 4 ] || [ $# -gt 5 ] ; then
    echo "Usage: ./run_uclanesl.sh [input_file] [output_file] [data_specs_file] [epsilon] [delta (optional)]"
    exit 1
fi

sampling_size=1000000
num_epochs=10
if [ $# -eq 4 ]; then
    python3 tf_gan.py --input_file=$1 --output_file=$2 --meta_file=$3 \
    --with_privacy=True --epsilon=$4 --sampling_size=$sampling_size \
    --num_epochs=$num_epochs
else
    python3 tf_gan.py --input_file=$1 --output_file=$2 --meta_file=$3 \
    --with_privacy=True --epsilon=$4 --delta=$5 --sampling_size=$sampling_size \
    --num_epochs=$num_epochs
fi