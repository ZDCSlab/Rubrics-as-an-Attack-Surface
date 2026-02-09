#!/bin/bash
export DEEPSEEK_API_KEY=sk-d6f66ed2606244c499c4c5a7e8775beb # set your DEEPSEEK_API_KEY here
# --- Optimization Settings ---

BATCH_SIZE=10 # Larger if possible

# --- Model Settings ---
MODEL=deepseek-chat # Set your model path
TEMP=0.0
TOP_P=1.0
MAX_TOKENS=1024

cd ./safety_task_double

for dataset in "AnthropicPku" "PkuRmb"
do

    # --- Path Configurations ---
    RUBRICS_DIR="./rubrics/${dataset}"
    RESULT_DIR="./results/deepseek/${dataset}"
    BENCH_DATA="./data/${dataset}/${dataset}-Bench/test.jsonl"
    TARGET_DATA="./data/${dataset}/${dataset}-Target/test.jsonl"

    python optimize/eval_api.py \
        --model_name $MODEL --batch_size $BATCH_SIZE --max_tokens $MAX_TOKENS \
        --prompt_path ./rubrics/${dataset}/selected_rubrics.jsonl \
        --data_path $BENCH_DATA \
        --output_dir ./results/${dataset} \
        --dataset_name bench --subset test

    python optimize/eval_api.py \
        --model_name $MODEL --batch_size $BATCH_SIZE --max_tokens $MAX_TOKENS \
        --prompt_path ./rubrics/${dataset}/selected_rubrics.jsonl \
        --data_path $TARGET_DATA \
        --output_dir ./results/${dataset} \
        --dataset_name target --subset test
done