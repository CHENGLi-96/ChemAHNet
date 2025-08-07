#!/bin/bash

# -------- Configuration --------
ENV_NAME=leash                     # Change to your actual conda environment name
SCRIPT=main_Majorgit.py              # Your main Python script
MODE=train                         # test mode
LOG_FILE=demo_Major_train.log            # output log file

# -------- Execution --------

echo "Activating conda environment: $ENV_NAME"
source ~/.bashrc
conda activate $ENV_NAME

echo "Running model in train mode..."
python $SCRIPT --local_rank 0 --$MODE \
--batch_size 64 --dropout=0.2 --num_heads 8 --num_layers 6 \
--embed_dim 256 --max_length 256 --output_dim 256 \
--prefix data --name tmp --epochs 250 \
> $LOG_FILE 2>&1

echo "Test finished. Output written to $LOG_FILE"

conda deactivate
