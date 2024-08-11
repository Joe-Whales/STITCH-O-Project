#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <model_path>"
    exit 1
fi

DATA_DIR=$1
MODEL_PATH=$2
@REM DATA_DIR="Preprocessing/data"

python segmentation/segment-orchards.py segmentation\segmentation-config.yaml "$DATA_DIR"

