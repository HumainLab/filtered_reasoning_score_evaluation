#!/bin/bash

# Submit Slurm jobs for 9 models x 6 datasets with zero-shot CoT prompt
# Parameters: temperature=0.7, top_p=0.95

API_URL="http://localhost:8009/jobs"

# Models (9 local models)
MODELS=(
    "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    "Qwen/Qwen2.5-Math-7B"
    "https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct"
    "https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507"
    "https://huggingface.co/google/gemma-7b"
    "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"
    "https://huggingface.co/microsoft/phi-4"
    "https://huggingface.co/microsoft/Phi-4-reasoning"
)

# Datasets (6 datasets)
DATASETS=(
    "gsm8k"
    "math500"
    "svamp"
    "aqua"
    "gpqa"
    "commonsense_qa"
)

# Parameters
TEMPERATURE=0.7
TOP_P=0.95
PROMPT_TYPE="cot"
BACKEND="slurm"

echo "Submitting jobs: 9 models x 6 datasets = 54 total jobs"
echo "Parameters: temperature=$TEMPERATURE, top_p=$TOP_P, prompt_type=$PROMPT_TYPE"
echo "============================================================"

job_count=0
success_count=0
fail_count=0

for model in "${MODELS[@]}"; do
    # Extract model name for display
    if [[ "$model" == https://* ]]; then
        model_name=$(echo "$model" | rev | cut -d'/' -f1 | rev)
    else
        model_name="$model"
    fi
    
    for dataset in "${DATASETS[@]}"; do
        job_count=$((job_count + 1))
        echo ""
        echo "[$job_count/54] Submitting: $model_name on $dataset"
        
        response=$(curl -s -X POST "$API_URL" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"$model\",
                \"dataset\": \"$dataset\",
                \"prompt_type\": \"$PROMPT_TYPE\",
                \"temperature\": $TEMPERATURE,
                \"top_p\": $TOP_P,
                \"backend\": \"$BACKEND\"
            }")
        
        # Check if job was submitted successfully
        if echo "$response" | grep -q '"job_id"'; then
            job_id=$(echo "$response" | grep -o '"job_id":"[^"]*"' | cut -d'"' -f4)
            slurm_jid=$(echo "$response" | grep -o '"slurm_jid":"[^"]*"' | cut -d'"' -f4)
            echo "  ✓ Success - Job ID: $job_id, Slurm JID: $slurm_jid"
            success_count=$((success_count + 1))
        else
            echo "  ✗ Failed - Response: $response"
            fail_count=$((fail_count + 1))
        fi
        
        # Small delay to avoid overwhelming the API
        sleep 0.5
    done
done

echo ""
echo "============================================================"
echo "Job submission complete!"
echo "Total: $job_count | Success: $success_count | Failed: $fail_count"
