#!/bin/bash
# main.sh

set -e

# ==================== Configurable Parameters ==================== #
CHATBOT_TRAINING_STEPS=${CHATBOT_TRAINING_STEPS:-10}

export TRAIN_BATCH_SIZE=60
export VAL_BATCH_SIZE=30
export TEST_FREQ=1

# ✅ Profile Generator Path
PROFILE_GENERATOR="/huggingface.co/Qwen/Qwen2.5-14B-Instruct" 


# ✅ Generation Retry Config
MAX_GENERATION_RETRIES=3

echo "=========================================="
echo "📋 Training Configuration"
echo "=========================================="
echo "  Chatbot Training Steps: $CHATBOT_TRAINING_STEPS"
echo "  Train Batch Size: $TRAIN_BATCH_SIZE"
echo "  Val Batch Size: $VAL_BATCH_SIZE"
echo "  Validation Freq: Every $TEST_FREQ steps"
echo "  14B Generator: $PROFILE_GENERATOR"
echo "  Max Generation Retries: $MAX_GENERATION_RETRIES"
echo "  GPU Allocation:"
echo "    - 14B Generator: GPU 0-7"
echo "    - vLLM Server: GPU 6-7"
echo "    - Chatbot Training: GPU 0-5"
echo "  vLLM Management Strategy:"
echo "    - Iteration 1-4: Keep running"
echo "    - Iteration 5: Close after training"
echo "=========================================="
echo ""

# ==================== Path Configuration ==================== #
export STORAGE_PATH="./outputs"
mkdir -p "$STORAGE_PATH/evaluation" "$STORAGE_PATH/models" "$STORAGE_PATH/prompt_data" "$STORAGE_PATH/temp" "$STORAGE_PATH/logs" "$STORAGE_PATH/temp_dialog"

Base_model="/huggingface.co/Qwen/Qwen2.5-14B-Instruct" 
Model_abbr="qwen2.5-14b-Instruct"

ROLLOUT_JSONL="${STORAGE_PATH}/temp_dialog/rollout_data.jsonl"

# ==================== Helper Functions ==================== #

# ✅ Function: Validate generated profile files
validate_generated_profiles() {
    local profile_file=$1
    local required_samples=$2
    
    echo ""
    echo "🔍 Validating generated profile file..."
    echo "   File: $profile_file"
    echo "   Requirement: $required_samples samples"
    
    if [ ! -f "$profile_file" ]; then
        echo "   ❌ File does not exist"
        return 1
    fi
    
    # Check file size
    local file_size=$(stat -f%z "$profile_file" 2>/dev/null || stat -c%s "$profile_file" 2>/dev/null)
    if [ "$file_size" -lt 100 ]; then
        echo "   ❌ File too small (${file_size} bytes)"
        return 1
    fi
    
    local line_count=$(wc -l < "$profile_file")
    echo "   📊 Actual line count: $line_count"
    
    if [ "$line_count" -lt "$required_samples" ]; then
        echo "   ⚠️  Insufficient samples ($line_count < $required_samples)"
        return 1
    fi
    
    # Validate JSON format (Sample check first 5 lines)
    echo "   🔍 Validating JSON format..."
    local invalid_count=0
    head -5 "$profile_file" | while read line; do
        if ! echo "$line" | python3 -m json.tool > /dev/null 2>&1; then
            invalid_count=$((invalid_count + 1))
        fi
    done
    
    if [ $invalid_count -gt 0 ]; then
        echo "   ⚠️  Found invalid JSON lines"
    fi
    
    echo "   ✅ Validation passed"
    return 0
}

# ✅ Function: Cleanup generation processes
cleanup_profile_generation_processes() {
    echo ""
    echo "🧹 Cleaning up all GPU-occupying processes..."
    
    bash scripts/force_cleanup_gpu.sh
    
    echo "✅ Cleanup completed"
}

# ✅ Function: Generate profiles (with retry)
generate_profiles_with_retry() {
    local analysis_file=$1
    local output_file=$2
    local num_samples=$3
    local max_retries=$4
    
    echo ""
    echo "=========================================="
    echo "🎨 Generate User Profiles (With Retry)"
    echo "=========================================="
    echo "   Analysis File: $analysis_file"
    echo "   Output File: $output_file"
    echo "   Num Samples: $num_samples"
    echo "   Max Retries: $max_retries"
    echo "=========================================="
    
    local retry_count=0
    local generation_success=false
    
    while [ $retry_count -lt $max_retries ]; do
        retry_count=$((retry_count + 1))
        
        echo ""
        echo "📦 Attempt $retry_count/$max_retries: Generating profiles..."
        
        if [ -f "$output_file" ]; then
            echo "   🗑️  Deleting old file: $output_file"
            rm -f "$output_file"
        fi
        
        cleanup_profile_generation_processes
        
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        
        echo "   🚀 Starting generation script..."
        
        if python3 ./utils/generate_profiles_parallel.py \
            --model_path "$PROFILE_GENERATOR" \
            --analysis_file "$analysis_file" \
            --behavior_library ./assets/client_action.jsonl \
            --output_file "$output_file" \
            --num_samples $num_samples \
            --num_workers 8; then
            
            echo "   ✅ Generation script execution completed"
            
            if validate_generated_profiles "$output_file" "$num_samples"; then
                echo ""
                echo "   ✅ Profile generation successful!"
                generation_success=true
                break
            else
                echo ""
                echo "   ⚠️  Validation failed, preparing to retry..."
            fi
        else
            echo "   ❌ Generation script execution failed"
        fi
        
        if [ $retry_count -lt $max_retries ]; then
            echo ""
            echo "⏳ Waiting 10 seconds before retry..."
            sleep 10
        fi
    done
    
    cleanup_profile_generation_processes
    
    if [ "$generation_success" = true ]; then
        echo ""
        echo "=========================================="
        echo "✅ Profile Generation Successful"
        echo "=========================================="
        return 0
    else
        echo ""
        echo "=========================================="
        echo "❌ Profile Generation Failed (Retried $max_retries times)"
        echo "=========================================="
        return 1
    fi
}

# ==================== Cleanup Existing Services (First Time Only) ==================== #
echo "🧹 Cleaning up existing vLLM services..."
bash scripts/cleanup_vllm.sh
sleep 2

echo ""
echo "=========================================="
echo "🚀 Starting Loop (Simplified)"
echo "=========================================="
echo "   Hardware: 8x A100 80G"
echo "   Max Turns: 15"
echo "   Total Iterations: 5"
echo "=========================================="
echo ""

# ==================== Iteration 1 ==================== #
echo "=========================================="
echo "🔄 Iteration 1 (Initialization)"
echo "=========================================="

echo ""
echo ">>> [Init] Generating Data (Random)..."

# ✅ Calculate data volume
TRAIN_SAMPLES=$((TRAIN_BATCH_SIZE * CHATBOT_TRAINING_STEPS))
ACTUAL_TRAINING_STEPS=$((CHATBOT_TRAINING_STEPS + 1))
TEST_SAMPLES=$VAL_BATCH_SIZE

echo "📊 Data Volume Calculation:"
echo "   Training Set: ${TRAIN_BATCH_SIZE} (batch) × ${CHATBOT_TRAINING_STEPS} (steps) = ${TRAIN_SAMPLES} samples"
echo "   Validation Set: ${VAL_BATCH_SIZE} samples (use all for each validation)"
echo "   Validation Count: ${ACTUAL_TRAINING_STEPS} times (step 0, 1, 2)"
echo "   Total Rollout: ${TRAIN_SAMPLES} (Train) + ${VAL_BATCH_SIZE} × ${ACTUAL_TRAINING_STEPS} (Val) = $((TRAIN_SAMPLES + VAL_BATCH_SIZE * ACTUAL_TRAINING_STEPS)) lines"

python ./utils/create_prompt_data.py \
    --mode chatbot \
    --train_samples $TRAIN_SAMPLES \
    --test_samples $TEST_SAMPLES \
    --out_dir ${STORAGE_PATH}/prompt_data \
    --txt_dir ./verl/trainer/config/format_prompt \
    --behavior_library ./assets/client_action.jsonl

if [ ! -f "${STORAGE_PATH}/prompt_data/train_chatbot.parquet" ]; then
    echo "❌ Error: Data generation failed for Iteration 1"
    exit 1
fi

echo ""
echo ">>> [Init] Training Chatbot v1..."

if [ -f "$ROLLOUT_JSONL" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    ARCHIVED_FILE="${STORAGE_PATH}/temp/rollout_data_iter_1_${TIMESTAMP}.jsonl"
    mv "$ROLLOUT_JSONL" "$ARCHIVED_FILE"
    echo "📦 Archived old rollout data to: $ARCHIVED_FILE"
fi

echo "📌 vLLM Server will remain running after training"
bash scripts/train_chatbot.sh \
    "$Base_model" \
    "$Base_model" \
    "${Model_abbr}_chatbot_v1" \
    1 \
    $ACTUAL_TRAINING_STEPS \
    true

CHATBOT_V1_CHECKPOINT="${STORAGE_PATH}/models/${Model_abbr}_chatbot_v1/actor/global_step_${CHATBOT_TRAINING_STEPS}"
if [ ! -d "$CHATBOT_V1_CHECKPOINT" ]; then
    echo "❌ Error: Chatbot v1 training failed"
    exit 1
fi

echo ""
echo ">>> [Init] Analyzing Chatbot mistakes..."
python3 ./utils/analyze_chatbot_mistakes.py \
    --rollout_file "$ROLLOUT_JSONL" \
    --output_file "${STORAGE_PATH}/chatbot_mistakes_analysis_step1.json"

echo ""
echo "✅ Iteration 1 completed!"
echo ""

# ==================== Loop: Iterations 2-5 ==================== #
for i in {2..5}; do
    prev=$((i-1))
    
    echo ""
    echo "=========================================="
    echo "🔄 Iteration $i"
    echo "=========================================="

    PREV_CHATBOT_MODEL="${STORAGE_PATH}/models/${Model_abbr}_chatbot_v${prev}/actor/global_step_${CHATBOT_TRAINING_STEPS}"
    
    if [ ! -d "$PREV_CHATBOT_MODEL" ]; then
        echo "❌ Error: Previous chatbot model not found"
        exit 1
    fi

    ANALYSIS_FILE="${STORAGE_PATH}/chatbot_mistakes_analysis_step${prev}.json"
    GENERATED_PROFILES="${STORAGE_PATH}/generated_profiles_iter_${i}.jsonl"
    
    # ✅ Recalculate data volume
    TRAIN_SAMPLES=$((TRAIN_BATCH_SIZE * CHATBOT_TRAINING_STEPS))
    ACTUAL_TRAINING_STEPS=$((CHATBOT_TRAINING_STEPS + 1))
    TEST_SAMPLES=$VAL_BATCH_SIZE
    TOTAL_SAMPLES=$((TRAIN_SAMPLES + TEST_SAMPLES))
    
    echo "📊 Data Volume Calculation:"
    echo "   Training Set: ${TRAIN_SAMPLES} samples"
    echo "   Validation Set: ${TEST_SAMPLES} samples"
    echo "   Total: ${TOTAL_SAMPLES} samples"
    
    # ✅ Generate profiles using model (with retry)
    if [ -f "$ANALYSIS_FILE" ]; then
        echo ""
        echo ">>> [Iter $i] Generating profiles..."
        
        # ✅ Call generation function with retry
        if generate_profiles_with_retry \
            "$ANALYSIS_FILE" \
            "$GENERATED_PROFILES" \
            "$TOTAL_SAMPLES" \
            "$MAX_GENERATION_RETRIES"; then
            
            echo ""
            echo ">>> [Iter $i] Generating Data (LLM-driven)..."
            python ./utils/create_prompt_data.py \
                --mode chatbot \
                --train_samples $TRAIN_SAMPLES \
                --test_samples $TEST_SAMPLES \
                --out_dir ${STORAGE_PATH}/prompt_data \
                --txt_dir ./verl/trainer/config/format_prompt \
                --behavior_library ./assets/client_action.jsonl \
                --generated_profiles "$GENERATED_PROFILES"
        else
            echo ""
            echo "⚠️ Generation failed after $MAX_GENERATION_RETRIES retries"
            echo "⚠️ Falling back to random generation"
            
            python ./utils/create_prompt_data.py \
                --mode chatbot \
                --train_samples $TRAIN_SAMPLES \
                --test_samples $TEST_SAMPLES \
                --out_dir ${STORAGE_PATH}/prompt_data \
                --txt_dir ./verl/trainer/config/format_prompt \
                --behavior_library ./assets/client_action.jsonl
        fi
    else
        echo ""
        echo "⚠️ Analysis file not found, using random generation"
        python ./utils/create_prompt_data.py \
            --mode chatbot \
            --train_samples $TRAIN_SAMPLES \
            --test_samples $TEST_SAMPLES \
            --out_dir ${STORAGE_PATH}/prompt_data \
            --txt_dir ./verl/trainer/config/format_prompt \
            --behavior_library ./assets/client_action.jsonl
    fi

    if [ ! -f "${STORAGE_PATH}/prompt_data/train_chatbot.parquet" ]; then
        echo "❌ Error: Data generation failed for Iteration $i"
        exit 1
    fi

    echo ""
    echo ">>> [Iter $i] Training Chatbot v$i..."
    
    if [ -f "$ROLLOUT_JSONL" ]; then
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        ARCHIVED_FILE="${STORAGE_PATH}/temp/rollout_data_iter_${i}_${TIMESTAMP}.jsonl"
        mv "$ROLLOUT_JSONL" "$ARCHIVED_FILE"
        echo "📦 Archived old rollout data to: $ARCHIVED_FILE"
    fi

    bash scripts/train_chatbot.sh \
        "$PREV_CHATBOT_MODEL" \
        "$Base_model" \
        "${Model_abbr}_chatbot_v${i}" \
        ${i} \
        $ACTUAL_TRAINING_STEPS \
        false

    CHATBOT_VI_CHECKPOINT="${STORAGE_PATH}/models/${Model_abbr}_chatbot_v${i}/actor/global_step_${CHATBOT_TRAINING_STEPS}"
    if [ ! -d "$CHATBOT_VI_CHECKPOINT" ]; then
        echo "❌ Error: Chatbot v${i} training failed"
        exit 1
    fi

    echo ""
    echo ">>> [Iter $i] Analyzing Chatbot mistakes..."
    python3 ./utils/analyze_chatbot_mistakes.py \
        --rollout_file "$ROLLOUT_JSONL" \
        --output_file "${STORAGE_PATH}/chatbot_mistakes_analysis_step${i}.json"

    echo ""
    echo "✅ Iteration $i completed!"
    echo ""
done

# ==================== Final Cleanup ==================== #
echo ""
echo "=========================================="
echo "🧹 Final cleanup of vLLM services..."
echo "=========================================="
bash scripts/cleanup_vllm.sh

echo ""
echo "=========================================="
echo "✅ Loop Completed!"
echo "=========================================="
echo ""
echo "📊 Training Results Summary:"
echo ""
for i in {1..5}; do
    MODEL_DIR="${STORAGE_PATH}/models/${Model_abbr}_chatbot_v${i}/actor/global_step_${CHATBOT_TRAINING_STEPS}"
    if [ -d "$MODEL_DIR" ]; then
        echo "  ✅ Chatbot v${i}: $MODEL_DIR"
    else
        echo "  ❌ Chatbot v${i}: Not Found"
    fi
done
echo ""

echo "📊 Generated Profile Files:"
echo ""
for i in {2..5}; do
    PROFILE_FILE="${STORAGE_PATH}/generated_profiles_iter_${i}.jsonl"
    if [ -f "$PROFILE_FILE" ]; then
        NUM_LINES=$(wc -l < "$PROFILE_FILE")
        echo "  ✅ Iteration ${i}: $PROFILE_FILE (${NUM_LINES} profiles)"
    else
        echo "  ❌ Iteration ${i}: Not Found"
    fi
done
echo ""

echo "📊 Analysis Report:"
echo ""
for i in {1..5}; do
    ANALYSIS_FILE="${STORAGE_PATH}/chatbot_mistakes_analysis_step${i}.json"
    if [ -f "$ANALYSIS_FILE" ]; then
        AVG_CR=$(python3 -c "import json; data=json.load(open('$ANALYSIS_FILE')); print(f\"{data.get('avg_cr', 0):.2%}\")" 2>/dev/null || echo "N/A")
        echo "  ✅ Step ${i}: CR=${AVG_CR}"
    else
        echo "  ❌ Step ${i}: Not Found"
    fi
done
echo ""

echo "=========================================="
echo "🎉 All training completed!"
echo "=========================================="