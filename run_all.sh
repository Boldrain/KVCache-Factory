#!/bin/bash

# 运行指令
# nohup ./run_all.sh > main.log 2>&1 &
# nohup ./run_all.sh quant_method nbits > main.log 2>&1 &


MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"
SAVE_DIR="./tmp_test"
MAX_CAPACITY=128
ATTN_IMPL="eager"

METHODS=("pyramidkv" "snapkv" "streamingllm" "h20")

# ===============================
# 处理 quant 参数
# ===============================

QUANT_METHOD=$1
NBITS=$2

USE_QUANT=false

if [ -n "$QUANT_METHOD" ]; then
    USE_QUANT=true
    echo "Quantization enabled: $QUANT_METHOD ($NBITS bits)"
else
    echo "Quantization disabled"
fi

mkdir -p logs

# ===============================
# 主循环
# ===============================

for METHOD in "${METHODS[@]}"
do
    echo "======================================="
    echo "Running method: $METHOD"
    echo "Start time: $(date)"
    echo "======================================="

    SAVE_PATH="$SAVE_DIR"

    if [ "$USE_QUANT" = true ]; then
        python run_longbench.py \
            --method "$METHOD" \
            --model_path "$MODEL_PATH" \
            --max_capacity_prompts "$MAX_CAPACITY" \
            --attn_implementation "$ATTN_IMPL" \
            --save_dir "$SAVE_PATH" \
            --use_cache True \
            --quant_method "$QUANT_METHOD" \
            --nbits "$NBITS" \
            > "logs/${METHOD}_${QUANT_METHOD}_${NBITS}.log" 2>&1
    else
        python run_longbench.py \
            --method "$METHOD" \
            --model_path "$MODEL_PATH" \
            --max_capacity_prompts "$MAX_CAPACITY" \
            --attn_implementation "$ATTN_IMPL" \
            --save_dir "$SAVE_PATH" \
            --use_cache True \
            > "logs/${METHOD}.log" 2>&1
    fi

    echo "Finished $METHOD at $(date)"
    echo ""
done

echo "All experiments completed at $(date)"
