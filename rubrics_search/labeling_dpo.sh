#!/bin/bash
export DEEPSEEK_API_KEY= # set your DEEPSEEK_API_KEY here
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # set your CUDA_VISIBLE_DEVICES here
# --- Optimization Settings ---

BATCH_SIZE=32 # Larger if possible

# --- Model Settings ---
MODEL=Qwen/Qwen3-14B # Set your model path
TEMP=0.0
TOP_P=1.0
MAX_TOKENS=4096

cd ./safety_task_double

for dataset in "AnthropicPku" 
do

    # --- Path Configurations ---
    RUBRICS_DIR="./rubrics/${dataset}"
    RESULT_DIR="./results/${dataset}"
    BENCH_DATA="./data/${dataset}/${dataset}-Bench-dpo/dpo_20k.jsonl"
    TARGET_DATA="./data/${dataset}/${dataset}-Target-dpo/dpo_20k.jsonl"

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python optimize/eval.py \
        --model_name $MODEL --batch_size $BATCH_SIZE --max_tokens $MAX_TOKENS \
        --prompt_path ./rubrics/${dataset}/selected_rubrics.jsonl \
        --data_path $BENCH_DATA \
        --output_dir ./results/${dataset} \
        --dataset_name bench-dpo --subset dpo_20k

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python optimize/eval.py \
        --model_name $MODEL --batch_size $BATCH_SIZE --max_tokens $MAX_TOKENS \
        --prompt_path ./rubrics/${dataset}/selected_rubrics.jsonl \
        --data_path $TARGET_DATA \
        --output_dir ./results/${dataset} \
        --dataset_name target-dpo --subset dpo_20k
done