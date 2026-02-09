#!/bin/bash
export DEEPSEEK_API_KEY= # set your DEEPSEEK_API_KEY here
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 # set your CUDA_VISIBLE_DEVICES here
# --- Optimization Settings ---
OPTIMIZE_TIMES=5
SAMPLE_SIZE=200
BATCH_SIZE=64 # Larger if possible
REFINE_NUMS=4

# --- Model Settings ---
MODEL=Qwen/Qwen3-14B # Set your model path
TEMP=0.0
TOP_P=1.0
MAX_TOKENS=4096

cd ./safety_task_double

for dataset in "AnthropicPku" "PkuRmb" "PkuAnthropic" "RmbPku"
do

    # --- Path Configurations ---
    RUBRICS_DIR="./rubrics/${dataset}"
    RESULT_DIR="./results/${dataset}"
    BENCH_DATA="./data/${dataset}/${dataset}-Bench/train.jsonl"
    TARGET_DATA="./data/${dataset}/${dataset}-Target/train.jsonl"

    # Execute the Python script
    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python optimize/main.py \
        --rubrics_dir "$RUBRICS_DIR" \
        --result_dir "$RESULT_DIR" \
        --bench_data "$BENCH_DATA" \
        --target_data "$TARGET_DATA" \
        --optimize_times $OPTIMIZE_TIMES \
        --sample_size $SAMPLE_SIZE \
        --batch_size $BATCH_SIZE \
        --refine_nums $REFINE_NUMS \
        --model_name "$MODEL" \
        --temp $TEMP \
        --top_p $TOP_P \
        --max_tokens $MAX_TOKENS

    python optimize/select_rubrics.py --folder ${dataset} --topk 10

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python optimize/eval.py \
        --model_name $MODEL --batch_size $BATCH_SIZE --max_tokens $MAX_TOKENS \
        --prompt_path ./rubrics/${dataset}/top_selected.jsonl \
        --data_path ./data/${dataset}/${dataset}-Target/val.jsonl \
        --output_dir ./results/${dataset} \
        --dataset_name target --subset val

    python optimize/select_final.py --folder ${dataset} --topk 10

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python optimize/eval.py \
        --model_name $MODEL --batch_size $BATCH_SIZE --max_tokens $MAX_TOKENS \
        --prompt_path ./rubrics/${dataset}/final.jsonl \
        --data_path ./data/${dataset}/${dataset}-Bench/val.jsonl \
        --output_dir ./results/${dataset} \
        --dataset_name bench --subset val

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python optimize/eval.py \
        --model_name $MODEL --batch_size $BATCH_SIZE --max_tokens $MAX_TOKENS \
        --prompt_path ./rubrics/${dataset}/final.jsonl \
        --data_path ./data/${dataset}/${dataset}-Target/test.jsonl \
        --output_dir ./results/${dataset} \
        --dataset_name target --subset test

    CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python optimize/eval.py \
        --model_name $MODEL --batch_size $BATCH_SIZE --max_tokens $MAX_TOKENS \
        --prompt_path ./rubrics/${dataset}/final.jsonl \
        --data_path ./data/${dataset}/${dataset}-Bench/test.jsonl \
        --output_dir ./results/${dataset} \
        --dataset_name bench --subset test

done