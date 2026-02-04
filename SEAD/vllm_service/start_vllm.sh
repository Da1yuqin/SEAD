#!/bin/bash
# SEAD/vllm_service/start_vllm.sh
# ================================================
# vLLM Server Startup Script (Supports GPU args + user_params)
# ================================================

set -e

MODEL_PATH=$1
RUN_ID=$2
STEP=$3
MODE=$4
PORT=${5:-5000}
GPU_IDS=${6:-"6,7"}  # ✅ Supports GPU args, default 6,7
USER_PARAMS_DIR=${7:-"./outputs/temp"}

source your/path/to/conda.sh
conda activate vllm


if [ -z "$MODEL_PATH" ] || [ -z "$RUN_ID" ] || [ -z "$STEP" ] || [ -z "$MODE" ]; then
    echo "❌ Error: Missing required arguments"
    echo "Usage: bash $0 <model_path> <run_id> <step> <mode> [port] [gpu_ids]"
    echo "  mode: call_client or call_client_evaluation"
    echo "  gpu_ids: e.g., '6,7' or '4,5' (default '6,7')"
    echo ""
    echo "Examples:"
    echo "  bash $0 /path/to/model run1 0 call_client 5000 '6,7'"
    echo "  bash $0 /path/to/model run2 0 call_client 5001 '4,5'"
    exit 1
fi


# Create log directory
LOG_DIR="./outputs/logs/vllm_server"
mkdir -p "$LOG_DIR"

# Create user_params directory (if not exists)
mkdir -p "$USER_PARAMS_DIR"

# Log file path
LOG_FILE="${LOG_DIR}/${MODE}_server_step${STEP}_port${PORT}.log"
PID_FILE="${LOG_DIR}/${MODE}_${RUN_ID}_step${STEP}_port${PORT}.pid"

echo "=========================================="
echo "🚀 Starting vLLM Server"
echo "=========================================="
echo "  Model:        $MODEL_PATH"
echo "  Mode:         $MODE"
echo "  Port:         $PORT"
echo "  GPU:          $GPU_IDS"
echo "  Step:         $STEP"
echo "  User Params:  $USER_PARAMS_DIR"
echo "  Log:          $LOG_FILE"
echo "=========================================="
echo ""

# ✅ Check user_params file (call_client mode only)
if [ "$MODE" = "call_client" ]; then
    echo "📂 Checking User Params file..."
    if [ -f "$USER_PARAMS_DIR/test_user_params.jsonl" ]; then
        PARAM_COUNT=$(wc -l < "$USER_PARAMS_DIR/test_user_params.jsonl")
        echo "   ✅ Found $PARAM_COUNT user_params"
    else
        echo "   ⚠️  test_user_params.jsonl not found"
        echo "   Path: $USER_PARAMS_DIR/test_user_params.jsonl"
        echo "   Service will use empty user_params"
    fi
    
    # Check/Create index file
    if [ ! -f "$USER_PARAMS_DIR/test_index.txt" ]; then
        echo "   ℹ️  Creating index file..."
        echo "0" > "$USER_PARAMS_DIR/test_index.txt"
    fi
    echo ""
fi

# ✅ Use provided GPU arguments
export CUDA_VISIBLE_DEVICES=$GPU_IDS  # Comment this out if limited GPUs available
# export CUDA_VISIBLE_DEVICES="0,1"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=0
export RAY_DEDUP_LOGS=0
export VLLM_SKIP_P2P_CHECK=1
export RAY_TMPDIR="/tmp/ray_${RUN_ID}_${PORT}"

echo "📊 Environment Variables:"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  VLLM_ATTENTION_BACKEND=$VLLM_ATTENTION_BACKEND"
echo "  VLLM_USE_V1=$VLLM_USE_V1"
echo "  VLLM_SKIP_P2P_CHECK=$VLLM_SKIP_P2P_CHECK"
echo "  RAY_TMPDIR=$RAY_TMPDIR"
echo ""

# Start vLLM Server
echo "🚀 Starting service..."
echo "   Logs will be saved to: $LOG_FILE"
echo ""

nohup python ./SEAD/vllm_service/vllm_server.py \
    --model_path "$MODEL_PATH" \
    --mode "$MODE" \
    --port "$PORT" \
    --user_params_dir "$USER_PARAMS_DIR" \
    --gpu_mem_util 0.70 \
    --max_model_len 8192 \
    --max_num_batched_tokens 200000 \
    --max_num_seqs 512 \
    --dtype bfloat16 \
    --max_gen_tokens 512 \
    --tensor_parallel_size 2 \
    --disable-custom-all-reduce \
    --enforce-eager \
    > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo $SERVER_PID > "$PID_FILE"

echo "✅ Service started"
echo "   Process PID: $SERVER_PID"
echo "   PID File: $PID_FILE"
echo ""

# Wait for service startup
echo "⏳ Waiting for service startup..."
MAX_WAIT=300
WAIT_COUNT=0

while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -s -f http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "✅ Service is ready!"
        echo ""
        
        # Print health check info
        HEALTH_INFO=$(curl -s http://localhost:${PORT}/health)
        echo "📊 Service Status:"
        echo "$HEALTH_INFO" | python3 -m json.tool 2>/dev/null || echo "$HEALTH_INFO"
        echo ""
        
        exit 0
    fi
    
    # Check if process is still running
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo ""
        echo "❌ Error: Service process has exited"
        echo "   Please check log file: $LOG_FILE"
        tail -50 "$LOG_FILE"
        exit 1
    fi
    
    echo "   Waiting... (${WAIT_COUNT}s / ${MAX_WAIT}s)"
    sleep 10
    WAIT_COUNT=$((WAIT_COUNT + 10))
done

echo ""
echo "❌ Error: Service startup timeout"
echo "   Please check log file: $LOG_FILE"
tail -50 "$LOG_FILE"
exit 1