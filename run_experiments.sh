#!/bin/bash

# Base configuration
CONFIG="config/train_shakespeare_char.py"

# Run experiments with different embedding sizes and seeds
for N_EMBD in 288 384 456 576; do
    BASE_NAME="CTXX2-d${N_EMBD}-bls2048-char"
    
    for SEED in 1 2 3; do
        RUN_NAME="${BASE_NAME}-${SEED}"
        echo "Starting run with embedding size ${N_EMBD}, seed ${SEED}, name: ${RUN_NAME}"
        python3 train.py $CONFIG \
            --wandb_run_name=$RUN_NAME \
            --block_type=cortex_x \
            --n_embd=$N_EMBD 
    done
done