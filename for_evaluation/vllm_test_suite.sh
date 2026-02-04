#!/bin/bash
# vllm_test_suite.sh - vLLM Test Suite Main Control Script (Supports Batch Testing Multiple Models)

set -e

# ==================== Configuration Section ====================

# User Simulator Configuration
# TODO: Replace with your actual model path
# Example: "Qwen/Qwen2.5-14B-Instruct" or "/path/to/your/user_simulator_model"
USER_SIM_MODEL="your/model/path"  # ⚠️ CHANGE THIS: Path to your user simulator model

NUM_USER_SIM_INSTANCES=2          # Number of user simulator instances to run in parallel
USER_SIM_BASE_PORT=5000           # Base port number (will use 5000, 5001, 5002, etc.)

# ✅ Chatbot Model List Configuration (Array Format)
# Format: "model_name|model_path"
# Example entries:
#   "qwen2.5-base|/path/to/qwen2.5-14b-base"
#   "llama3-finetuned|/path/to/llama3-8b-finetuned"
declare -a CHATBOT_MODELS=(
    "qwen2.5-SEAD|./outputs/models/qwen2.5-14b-Instruct_chatbot_v4/actor/global_step_10"  # ⚠️ CHANGE THIS
    # Add more models here:
    # "model_name_2|/path/to/model_2"
    # "model_name_3|/path/to/model_3"
)

# GPU IDs for chatbot inference (comma-separated)
# Example: "0,1,2,3" for using GPUs 0-3
CHATBOT_GPUS="0,1,2,3,4,5"  # ⚠️ CHANGE THIS: Adjust based on your available GPUs

# Test Configuration
TEST_DATA="./outputs/evaluation/test_set/test_chatbot.parquet"  # ⚠️ CHANGE THIS: Path to your test dataset
N_SAMPLES=1000                    # Number of samples to test
BATCH_SIZE=64                     # Batch size for inference
MAX_NEW_TOKENS=512                # Maximum number of tokens to generate
USER_PARAMS_DIR="./outputs/evaluation/test_set/user_param"  # ⚠️ CHANGE THIS: Directory containing user parameters

# ✅ Batch Test Configuration
CLEANUP_BETWEEN_TESTS=true        # Whether to cleanup environment between each model test
WAIT_BETWEEN_TESTS=60             # Wait time (seconds) between tests

# ==================================================

# Color Definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Print Functions
print_header() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
    echo ""
}

print_step() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_model_header() {
    echo ""
    echo -e "${MAGENTA}╔════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${MAGENTA}║                                                                ║${NC}"
    echo -e "${MAGENTA}║  $1${NC}"
    echo -e "${MAGENTA}║                                                                ║${NC}"
    echo -e "${MAGENTA}╚════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# ==================== Function Definitions ====================

# Cleanup Environment
cleanup_environment() {
    print_step "🧹 Cleaning Up Environment"
    
    echo "1️⃣ Stopping vLLM processes..."
    VLLM_PIDS=$(ps aux | grep "vllm_server.py" | grep -v grep | awk '{print $2}')
    if [ -n "$VLLM_PIDS" ]; then
        echo "   Found vLLM processes: $VLLM_PIDS"
        echo "$VLLM_PIDS" | xargs kill -9 2>/dev/null || true
        print_success "Stopped vLLM processes"
    else
        echo "   ℹ️  No running vLLM processes"
    fi
    sleep 2
    
    echo ""
    echo "2️⃣ Stopping Ray processes..."
    ray stop --force 2>/dev/null || true
    sleep 3
    print_success "Stopped Ray processes"
    
    echo ""
    echo "3️⃣ Cleaning Ray temporary files..."
    RAY_DIRS=$(ls -d /tmp/ray_* 2>/dev/null || true)
    if [ -n "$RAY_DIRS" ]; then
        rm -rf /tmp/ray_* 2>/dev/null || true
        print_success "Cleaned Ray temporary files"
    else
        echo "   ℹ️  No Ray temporary files"
    fi
    
    echo ""
    echo "4️⃣ Cleaning PID files..."
    if [ -d "./outputs/logs/vllm_server" ]; then
        rm -f ./outputs/logs/vllm_server/*.pid 2>/dev/null || true
        print_success "Cleaned PID files"
    fi
    
    echo ""
    echo "5️⃣ Checking port usage..."
    for port in 5000 5001 5002 5003; do
        PID=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$PID" ]; then
            echo "   Releasing port $port (PID: $PID)..."
            kill -9 $PID 2>/dev/null || true
        fi
    done
    print_success "Port check completed"
    
    echo ""
    echo "6️⃣ GPU Status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F', ' '{printf "   GPU %s: %s - Using %s MB / %s MB\n", $1, $2, $3, $4}'
    
    print_success "Environment cleanup completed"
}

# Start User Simulator Instances
start_user_simulators() {
    print_step "🚀 Starting User Simulator Instances"
    
    echo "Configuration:"
    echo "  Model Path:       $USER_SIM_MODEL"
    echo "  Num Instances:    $NUM_USER_SIM_INSTANCES"
    echo "  Base Port:        $USER_SIM_BASE_PORT"
    echo "  User Params Dir:  $USER_PARAMS_DIR"
    echo ""
    
    echo "📂 Checking User Params files..."
    if [ ! -d "$USER_PARAMS_DIR" ]; then
        print_error "User Params directory does not exist: $USER_PARAMS_DIR"
        echo "   Please run create_data first to generate user_params"
        exit 1
    fi
    
    if [ ! -f "$USER_PARAMS_DIR/test_user_params.jsonl" ]; then
        print_error "Cannot find test_user_params.jsonl"
        echo "   Path: $USER_PARAMS_DIR/test_user_params.jsonl"
        exit 1
    fi
    
    PARAM_COUNT=$(wc -l < "$USER_PARAMS_DIR/test_user_params.jsonl")
    print_success "Found $PARAM_COUNT user_params"
    
    if [ ! -d "$USER_SIM_MODEL" ]; then
        print_error "Model path does not exist: $USER_SIM_MODEL"
        exit 1
    fi
    
    if [ ! -f "./SEAD/vllm_service/start_vllm.sh" ]; then
        print_error "Cannot find ./SEAD/vllm_service/start_vllm.sh script"
        exit 1
    fi
    
    # GPU assignments for each user simulator instance
    # ⚠️ CHANGE THIS: Adjust GPU assignments based on your hardware
    # Example: If you have 8 GPUs, you might use:
    #   declare -a GPU_ASSIGNMENTS=("0,1" "2,3" "4,5" "6,7")
    declare -a GPU_ASSIGNMENTS=("6,7" "4,5")
    
    for i in $(seq 0 $((NUM_USER_SIM_INSTANCES - 1))); do
        PORT=$((USER_SIM_BASE_PORT + i))
        RUN_ID="usersim_${i}"
        INSTANCE_GPU_IDS=${GPU_ASSIGNMENTS[$i]}
        
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Starting Instance $((i + 1))/$NUM_USER_SIM_INSTANCES"
        echo "  Port:       $PORT"
        echo "  GPU:        $INSTANCE_GPU_IDS"
        echo "  Run ID:     $RUN_ID"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        
        bash ./SEAD/vllm_service/start_vllm.sh \
            "$USER_SIM_MODEL" \
            "$RUN_ID" \
            "0" \
            "call_client" \
            "$PORT" \
            "$INSTANCE_GPU_IDS"
        
        if [ $? -eq 0 ]; then
            print_success "Instance $((i + 1)) started successfully"
        else
            print_error "Instance $((i + 1)) failed to start"
            echo "   Please check logs: ./outputs/logs/vllm_server/call_client_server_step0_port${PORT}.log"
            exit 1
        fi
        
        if [ $i -lt $((NUM_USER_SIM_INSTANCES - 1)) ]; then
            echo ""
            echo "⏳ Waiting 30 seconds before starting next instance..."
            sleep 30
        fi
    done
    
    echo ""
    print_step "📊 Verifying User Simulator Instances"
    
    ALL_READY=true
    READY_URLS=""
    
    for i in $(seq 0 $((NUM_USER_SIM_INSTANCES - 1))); do
        PORT=$((USER_SIM_BASE_PORT + i))
        INSTANCE_GPU_IDS=${GPU_ASSIGNMENTS[$i]}
        
        if curl -s -f http://localhost:${PORT}/health > /dev/null 2>&1; then
            print_success "Instance $((i + 1)) (Port $PORT, GPU $INSTANCE_GPU_IDS) - Ready"
            READY_URLS="$READY_URLS http://localhost:$PORT"
        else
            print_error "Instance $((i + 1)) (Port $PORT, GPU $INSTANCE_GPU_IDS) - Not Ready"
            ALL_READY=false
        fi
    done
    
    if [ "$ALL_READY" = true ]; then
        echo ""
        print_success "All User Simulator instances are ready"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "User Simulator URLs:"
        echo "$READY_URLS"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        
        export USER_SIM_URLS="$READY_URLS"
    else
        print_error "Some instances failed to start"
        exit 1
    fi
}

# ✅ Run Single Model Test
run_single_test() {
    local MODEL_NAME=$1
    local MODEL_PATH=$2
    
    print_step "🧪 Testing Model: $MODEL_NAME"
    
    echo "Configuration:"
    echo "  Chatbot Model:    $MODEL_PATH"
    echo "  Model Name:       $MODEL_NAME"
    echo "  Test Data:        $TEST_DATA"
    echo "  Num Samples:      $N_SAMPLES"
    echo "  Batch Size:       $BATCH_SIZE"
    echo "  Chatbot GPUs:     $CHATBOT_GPUS"
    echo "  User Sim URLs:    $USER_SIM_URLS"
    echo ""
    
    # Check model path
    if [ ! -d "$MODEL_PATH" ]; then
        print_error "Chatbot model path does not exist: $MODEL_PATH"
        return 1
    fi
    
    # Check User Simulator
    echo "🔍 Checking User Simulator status..."
    ALL_READY=true
    
    for url in $USER_SIM_URLS; do
        if curl -s -f ${url}/health > /dev/null 2>&1; then
            print_success "$url - Ready"
        else
            print_error "$url - Cannot connect"
            ALL_READY=false
        fi
    done
    
    if [ "$ALL_READY" = false ]; then
        print_error "User Simulator not ready"
        return 1
    fi
    
    echo ""
    print_step "🚀 Starting Test: $MODEL_NAME"
    
    CUDA_VISIBLE_DEVICES=$CHATBOT_GPUS \
    VLLM_USE_V1=0 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    python ./for_evaluation/Baseline_test_local_models_vllm.py \
        --model_path "$MODEL_PATH" \
        --model_name "$MODEL_NAME" \
        --test_data "$TEST_DATA" \
        --n_samples $N_SAMPLES \
        --batch_size $BATCH_SIZE \
        --max_new_tokens $MAX_NEW_TOKENS \
        --user_sim_urls $USER_SIM_URLS \
        --user_params_dir "$USER_PARAMS_DIR" \
        --tensor_parallel_size 4 \
        --gpu_memory_utilization 0.9
    
    if [ $? -eq 0 ]; then
        echo ""
        print_success "Model $MODEL_NAME test completed"
        echo ""
        echo "Result files:"
        echo "  - ./outputs/evaluation/${MODEL_NAME}_dialogues.jsonl"
        echo "  - ./outputs/evaluation/${MODEL_NAME}_results.json"
        echo ""
        return 0
    else
        print_error "Model $MODEL_NAME test failed"
        return 1
    fi
}

# ✅ Batch Test All Models
run_batch_tests() {
    print_header "🚀 Starting Batch Test for ${#CHATBOT_MODELS[@]} Models"
    
    # Record start time
    BATCH_START_TIME=$(date +%s)
    
    # Create test summary file
    SUMMARY_FILE="./outputs/evaluation/batch_test_summary_$(date +%Y%m%d_%H%M%S).txt"
    mkdir -p ./outputs/evaluation
    
    echo "Batch Test Summary" > "$SUMMARY_FILE"
    echo "Start Time: $(date)" >> "$SUMMARY_FILE"
    echo "Total Models: ${#CHATBOT_MODELS[@]}" >> "$SUMMARY_FILE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    
    # Statistics variables
    SUCCESS_COUNT=0
    FAILED_COUNT=0
    declare -a FAILED_MODELS
    
    # Loop through each model
    for idx in "${!CHATBOT_MODELS[@]}"; do
        MODEL_CONFIG="${CHATBOT_MODELS[$idx]}"
        MODEL_NAME=$(echo "$MODEL_CONFIG" | cut -d'|' -f1)
        MODEL_PATH=$(echo "$MODEL_CONFIG" | cut -d'|' -f2)
        
        CURRENT_NUM=$((idx + 1))
        TOTAL_NUM=${#CHATBOT_MODELS[@]}
        
        print_model_header "Test Progress: [$CURRENT_NUM/$TOTAL_NUM] - $MODEL_NAME"
        
        echo "Model Info:" | tee -a "$SUMMARY_FILE"
        echo "  Index: $CURRENT_NUM/$TOTAL_NUM" | tee -a "$SUMMARY_FILE"
        echo "  Name: $MODEL_NAME" | tee -a "$SUMMARY_FILE"
        echo "  Path: $MODEL_PATH" | tee -a "$SUMMARY_FILE"
        echo "" | tee -a "$SUMMARY_FILE"
        
        # Record single model start time
        MODEL_START_TIME=$(date +%s)
        
        # Run test
        if run_single_test "$MODEL_NAME" "$MODEL_PATH"; then
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            MODEL_END_TIME=$(date +%s)
            MODEL_DURATION=$((MODEL_END_TIME - MODEL_START_TIME))
            
            echo "✅ Success - $MODEL_NAME (Duration: ${MODEL_DURATION}s)" >> "$SUMMARY_FILE"
            print_success "Model $MODEL_NAME test succeeded (Duration: ${MODEL_DURATION}s)"
        else
            FAILED_COUNT=$((FAILED_COUNT + 1))
            FAILED_MODELS+=("$MODEL_NAME")
            
            echo "❌ Failed - $MODEL_NAME" >> "$SUMMARY_FILE"
            print_error "Model $MODEL_NAME test failed"
        fi
        
        echo "" >> "$SUMMARY_FILE"
        
        # If not the last model, cleanup and wait
        if [ $CURRENT_NUM -lt $TOTAL_NUM ]; then
            if [ "$CLEANUP_BETWEEN_TESTS" = true ]; then
                echo ""
                print_step "🧹 Cleaning environment, preparing for next model"
                cleanup_environment
                
                echo ""
                echo "⏳ Waiting ${WAIT_BETWEEN_TESTS} seconds before continuing..."
                sleep $WAIT_BETWEEN_TESTS
                
                # Restart User Simulator
                start_user_simulators
            else
                echo ""
                echo "⏳ Waiting 10 seconds before continuing..."
                sleep 10
            fi
        fi
    done
    
    # Calculate total duration
    BATCH_END_TIME=$(date +%s)
    BATCH_DURATION=$((BATCH_END_TIME - BATCH_START_TIME))
    BATCH_DURATION_MIN=$((BATCH_DURATION / 60))
    
    # Generate final summary
    echo "" >> "$SUMMARY_FILE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >> "$SUMMARY_FILE"
    echo "Test Completion Summary" >> "$SUMMARY_FILE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >> "$SUMMARY_FILE"
    echo "End Time: $(date)" >> "$SUMMARY_FILE"
    echo "Total Duration: ${BATCH_DURATION}s (${BATCH_DURATION_MIN}min)" >> "$SUMMARY_FILE"
    echo "Success: $SUCCESS_COUNT" >> "$SUMMARY_FILE"
    echo "Failed: $FAILED_COUNT" >> "$SUMMARY_FILE"
    
    if [ $FAILED_COUNT -gt 0 ]; then
        echo "" >> "$SUMMARY_FILE"
        echo "Failed Models:" >> "$SUMMARY_FILE"
        for failed_model in "${FAILED_MODELS[@]}"; do
            echo "  - $failed_model" >> "$SUMMARY_FILE"
        done
    fi
    
    # Display final results
    echo ""
    print_header "📊 Batch Test Completed"
    
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}Test Summary${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo "  Total Models: ${#CHATBOT_MODELS[@]}"
    echo "  Success: $SUCCESS_COUNT"
    echo "  Failed: $FAILED_COUNT"
    echo "  Total Duration: ${BATCH_DURATION}s (${BATCH_DURATION_MIN}min)"
    echo ""
    echo "  Summary File: $SUMMARY_FILE"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if [ $FAILED_COUNT -gt 0 ]; then
        echo ""
        print_warning "The following models failed:"
        for failed_model in "${FAILED_MODELS[@]}"; do
            echo "  - $failed_model"
        done
    fi
    
    echo ""
    
    # Return failure count as exit code
    return $FAILED_COUNT
}

# Stop All Services
stop_services() {
    print_step "🛑 Stopping All Services"
    
    echo "Stopping User Simulator instances..."
    for i in $(seq 0 $((NUM_USER_SIM_INSTANCES - 1))); do
        PORT=$((USER_SIM_BASE_PORT + i))
        PID_FILE="./outputs/logs/vllm_server/call_client_usersim_${i}_step0_port${PORT}.pid"
        
        if [ -f "$PID_FILE" ]; then
            PID=$(cat "$PID_FILE")
            if kill -0 $PID 2>/dev/null; then
                kill -9 $PID 2>/dev/null || true
                echo "  Stopped instance $((i + 1)) (PID: $PID)"
            fi
            rm -f "$PID_FILE"
        fi
    done
    
    cleanup_environment
    print_success "All services stopped"
}

# Show Status
show_status() {
    print_step "📊 Service Status"
    
    echo "User Simulator Instances:"
    declare -a GPU_ASSIGNMENTS=("6,7" "4,5")  # ⚠️ Should match the assignments in start_user_simulators
    
    for i in $(seq 0 $((NUM_USER_SIM_INSTANCES - 1))); do
        PORT=$((USER_SIM_BASE_PORT + i))
        INSTANCE_GPU_IDS=${GPU_ASSIGNMENTS[$i]}
        
        if curl -s -f http://localhost:${PORT}/health > /dev/null 2>&1; then
            print_success "Instance $((i + 1)) (Port $PORT, GPU $INSTANCE_GPU_IDS) - Running"
        else
            print_error "Instance $((i + 1)) (Port $PORT, GPU $INSTANCE_GPU_IDS) - Not Running"
        fi
    done
    
    echo ""
    echo "Configured Model List:"
    for idx in "${!CHATBOT_MODELS[@]}"; do
        MODEL_CONFIG="${CHATBOT_MODELS[$idx]}"
        MODEL_NAME=$(echo "$MODEL_CONFIG" | cut -d'|' -f1)
        echo "  $((idx + 1)). $MODEL_NAME"
    done
    
    echo ""
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F', ' '{printf "  GPU %s: %s - Utilization %s%% - Memory %s/%s MB\n", $1, $2, $3, $4, $5}'
}

# ==================== Main Menu ====================

show_menu() {
    print_header "vLLM Test Suite - Main Menu (Supports Batch Testing)"
    
    echo "Please select an option:"
    echo ""
    echo "  1) Batch test all models (Recommended)"
    echo "  2) Full workflow (Cleanup → Start → Batch Test)"
    echo "  3) Cleanup environment only"
    echo "  4) Start User Simulator only"
    echo "  5) Stop all services"
    echo "  6) Show status"
    echo "  7) Show model list"
    echo "  8) Exit"
    echo ""
    read -p "Enter option [1-8]: " choice
    
    case $choice in
        1)
            # Build USER_SIM_URLS
            USER_SIM_URLS=""
            for i in $(seq 0 $((NUM_USER_SIM_INSTANCES - 1))); do
                PORT=$((USER_SIM_BASE_PORT + i))
                USER_SIM_URLS="$USER_SIM_URLS http://localhost:$PORT"
            done
            export USER_SIM_URLS
            run_batch_tests
            ;;
        2)
            cleanup_environment
            start_user_simulators
            run_batch_tests
            ;;
        3)
            cleanup_environment
            ;;
        4)
            start_user_simulators
            ;;
        5)
            stop_services
            ;;
        6)
            show_status
            ;;
        7)
            echo ""
            echo "Configured Model List (Total: ${#CHATBOT_MODELS[@]}):"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            for idx in "${!CHATBOT_MODELS[@]}"; do
                MODEL_CONFIG="${CHATBOT_MODELS[$idx]}"
                MODEL_NAME=$(echo "$MODEL_CONFIG" | cut -d'|' -f1)
                MODEL_PATH=$(echo "$MODEL_CONFIG" | cut -d'|' -f2)
                echo ""
                echo "$((idx + 1)). $MODEL_NAME"
                echo "   Path: $MODEL_PATH"
            done
            echo ""
            ;;
        8)
            echo "Exiting"
            exit 0
            ;;
        *)
            print_error "Invalid option"
            exit 1
            ;;
    esac
}

# ==================== Command Line Arguments ====================

if [ $# -eq 0 ]; then
    show_menu
else
    case $1 in
        cleanup|clean)
            cleanup_environment
            ;;
        start)
            start_user_simulators
            ;;
        batch|test-all)
            USER_SIM_URLS=""
            for i in $(seq 0 $((NUM_USER_SIM_INSTANCES - 1))); do
                PORT=$((USER_SIM_BASE_PORT + i))
                USER_SIM_URLS="$USER_SIM_URLS http://localhost:$PORT"
            done
            export USER_SIM_URLS
            run_batch_tests
            ;;
        stop)
            stop_services
            ;;
        status)
            show_status
            ;;
        all|full)
            cleanup_environment
            start_user_simulators
            run_batch_tests
            ;;
        list)
            echo ""
            echo "Configured Model List (Total: ${#CHATBOT_MODELS[@]}):"
            for idx in "${!CHATBOT_MODELS[@]}"; do
                MODEL_CONFIG="${CHATBOT_MODELS[$idx]}"
                MODEL_NAME=$(echo "$MODEL_CONFIG" | cut -d'|' -f1)
                echo "  $((idx + 1)). $MODEL_NAME"
            done
            echo ""
            ;;
        help|-h|--help)
            echo "Usage: bash $0 [command]"
            echo ""
            echo "Commands:"
            echo "  cleanup/clean  - Cleanup environment"
            echo "  start          - Start User Simulator"
            echo "  batch/test-all - Batch test all models"
            echo "  stop           - Stop all services"
            echo "  status         - Show status"
            echo "  list           - Show model list"
            echo "  all/full       - Full workflow (Cleanup + Start + Batch Test)"
            echo "  help           - Show this help"
            echo ""
            echo "Run without arguments to show interactive menu"
            ;;
        *)
            print_error "Unknown command: $1"
            echo "Run 'bash $0 help' for help"
            exit 1
            ;;
    esac
fi