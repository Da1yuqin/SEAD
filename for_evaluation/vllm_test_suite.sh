#!/bin/bash
# vllm_test_suite.sh - vLLM Test Suite Main Script (Supports Batch Testing of Multiple Models)

# set -e  # Do not use set -e, allow continuing after single model failure

# ==================== Configuration ====================

# User Simulator Configuration
USER_SIM_MODEL="your/path/to/huggingface.co/Qwen/Qwen2.5-14B-Instruct"  # TODO: Replace with your user simulator model path (e.g., Qwen2.5-14B-Instruct)
NUM_USER_SIM_INSTANCES=2
USER_SIM_BASE_PORT=5000

# Chatbot Model List Configuration (array format)
# Format: "model_name|model_path"
BASE_MODELS_DIR="your/path/to/outputs/models"  # TODO: Replace with your trained models output directory
declare -a CHATBOT_MODELS=(
    "qwen2.5-SEAD|./outputs/models/qwen2.5-14b-Instruct_chatbot_v4/actor/global_step_10"  # ⚠️ CHANGE THIS
    # Add more models here:
    # "model_name_2|/path/to/model_2"
    # "model_name_3|/path/to/model_3"
)


CHATBOT_GPUS="0,1,2,3,4,5"

# Test Configuration
TEST_DATA="./outputs/evaluation/test_set/test_chatbot.parquet"
N_SAMPLES=1000
BATCH_SIZE=30
MAX_NEW_TOKENS=512
USER_PARAMS_DIR="./outputs/evaluation/test_set/user_param"

# Batch Test Configuration
CLEANUP_BETWEEN_TESTS=true  # Whether to clean up environment after each model test
WAIT_BETWEEN_TESTS=60       # Wait time between tests (seconds)

# ==================================================

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Print functions
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
    echo -e "${GREEN}[OK] $1${NC}"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}[WARN] $1${NC}"
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

# Clean up environment
cleanup_environment() {
    print_step "Cleaning up environment"
    
    echo "1. Stopping vLLM processes..."
    VLLM_PIDS=$(ps aux | grep "vllm_server.py" | grep -v grep | awk '{print $2}')
    if [ -n "$VLLM_PIDS" ]; then
        echo "   Found vLLM processes: $VLLM_PIDS"
        echo "$VLLM_PIDS" | xargs kill -9 2>/dev/null || true
        print_success "Stopped vLLM processes"
    else
        echo "   No running vLLM processes"
    fi
    sleep 2
    
    echo ""
    echo "2. Stopping Ray processes..."
    ray stop --force 2>/dev/null || true
    sleep 3
    print_success "Stopped Ray processes"
    
    echo ""
    echo "3. Cleaning Ray temporary files..."
    RAY_DIRS=$(ls -d /tmp/ray_* 2>/dev/null || true)
    if [ -n "$RAY_DIRS" ]; then
        rm -rf /tmp/ray_* 2>/dev/null || true
        print_success "Cleaned Ray temporary files"
    else
        echo "   No Ray temporary files"
    fi
    
    echo ""
    echo "4. Cleaning PID files..."
    if [ -d "./outputs/logs/vllm_server" ]; then
        rm -f ./outputs/logs/vllm_server/*.pid 2>/dev/null || true
        print_success "Cleaned PID files"
    fi
    
    echo ""
    echo "5. Checking port usage..."
    for port in 5000 5001 5002 5003; do
        PID=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$PID" ]; then
            echo "   Releasing port $port (PID: $PID)..."
            kill -9 $PID 2>/dev/null || true
        fi
    done
    print_success "Port check complete"
    
    echo ""
    echo "6. GPU status:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F', ' '{printf "   GPU %s: %s - Using %s MB / %s MB\n", $1, $2, $3, $4}'
    
    print_success "Environment cleanup complete"
}

# Start User Simulator instances
start_user_simulators() {
    print_step "Starting User Simulator instances"
    
    echo "Configuration info:"
    echo "  Model path:      $USER_SIM_MODEL"
    echo "  Instance count:  $NUM_USER_SIM_INSTANCES"
    echo "  Base port:       $USER_SIM_BASE_PORT"
    echo "  User Params:     $USER_PARAMS_DIR"
    echo ""
    
    echo "Checking User Params files..."
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
    
    declare -a GPU_ASSIGNMENTS=("6" "7")
    
    for i in $(seq 0 $((NUM_USER_SIM_INSTANCES - 1))); do
        PORT=$((USER_SIM_BASE_PORT + i))
        RUN_ID="eval_usersim_${i}"
        INSTANCE_GPU_IDS=${GPU_ASSIGNMENTS[$i]}
        
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Starting instance $((i + 1))/$NUM_USER_SIM_INSTANCES"
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
            echo "   Please check log: ./outputs/logs/vllm_server/call_client_server_step0_port${PORT}.log"
            exit 1
        fi
        
        if [ $i -lt $((NUM_USER_SIM_INSTANCES - 1)) ]; then
            echo ""
            echo "Waiting 30 seconds before starting next instance..."
            sleep 30
        fi
    done
    
    echo ""
    print_step "Verifying User Simulator instances"
    
    ALL_READY=true
    READY_URLS=""
    
    for i in $(seq 0 $((NUM_USER_SIM_INSTANCES - 1))); do
        PORT=$((USER_SIM_BASE_PORT + i))
        INSTANCE_GPU_IDS=${GPU_ASSIGNMENTS[$i]}
        
        if curl -s -f http://localhost:${PORT}/health > /dev/null 2>&1; then
            print_success "Instance $((i + 1)) (port $PORT, GPU $INSTANCE_GPU_IDS) - Normal"
            READY_URLS="$READY_URLS http://localhost:$PORT"
        else
            print_error "Instance $((i + 1)) (port $PORT, GPU $INSTANCE_GPU_IDS) - Abnormal"
            ALL_READY=false
        fi
    done
    
    if [ "$ALL_READY" = true ]; then
        echo ""
        print_success "All User Simulator instances ready"
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

# Run single model test
run_single_test() {
    local MODEL_NAME=$1
    local MODEL_PATH=$2
    
    print_step "Testing model: $MODEL_NAME"
    
    echo "Configuration info:"
    echo "  Chatbot model: $MODEL_PATH"
    echo "  Model name:    $MODEL_NAME"
    echo "  Test data:     $TEST_DATA"
    echo "  Num samples:   $N_SAMPLES"
    echo "  Batch size:    $BATCH_SIZE"
    echo "  Chatbot GPU:   $CHATBOT_GPUS"
    echo "  User Sim:      $USER_SIM_URLS"
    echo ""
    
    # Check model path
    if [ ! -d "$MODEL_PATH" ]; then
        print_error "Chatbot model path does not exist: $MODEL_PATH"
        return 1
    fi
    
    # Check User Simulator
    echo "Checking User Simulator status..."
    ALL_READY=true
    
    for url in $USER_SIM_URLS; do
        if curl -s -f ${url}/health > /dev/null 2>&1; then
            print_success "$url - Normal"
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
    print_step "Starting test: $MODEL_NAME"
    
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
        --gpu_memory_utilization 0.9 \
        --vllm_model_path "$USER_SIM_MODEL"
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo ""
        print_success "Model $MODEL_NAME test completed"
        echo ""
        echo "Result files:"
        echo "  - ./outputs/evaluation/${MODEL_NAME}_dialogues.jsonl"
        echo "  - ./outputs/evaluation/${MODEL_NAME}_results.json"
        echo ""
        return 0
    else
        print_error "Model $MODEL_NAME test failed (exit_code=$exit_code)"
        return 1
    fi
}

# Batch test all models
run_batch_tests() {
    print_header "Starting batch testing of ${#CHATBOT_MODELS[@]} models"
    
    # Record start time
    BATCH_START_TIME=$(date +%s)
    
    # Create test summary file
    SUMMARY_FILE="./outputs/evaluation/batch_test_summary_$(date +%Y%m%d_%H%M%S).txt"
    mkdir -p ./outputs/evaluation
    
    echo "Batch Test Summary" > "$SUMMARY_FILE"
    echo "Start time: $(date)" >> "$SUMMARY_FILE"
    echo "Total models: ${#CHATBOT_MODELS[@]}" >> "$SUMMARY_FILE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"
    
    # Statistics variables
    SUCCESS_COUNT=0
    FAILED_COUNT=0
    declare -a FAILED_MODELS
    
    # Test each model in loop
    for idx in "${!CHATBOT_MODELS[@]}"; do
        MODEL_CONFIG="${CHATBOT_MODELS[$idx]}"
        MODEL_NAME=$(echo "$MODEL_CONFIG" | cut -d'|' -f1)
        MODEL_PATH=$(echo "$MODEL_CONFIG" | cut -d'|' -f2)
        
        CURRENT_NUM=$((idx + 1))
        TOTAL_NUM=${#CHATBOT_MODELS[@]}
        
        print_model_header "Test progress: [$CURRENT_NUM/$TOTAL_NUM] - $MODEL_NAME"
        
        echo "Model info:" | tee -a "$SUMMARY_FILE"
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
            
            echo "SUCCESS - $MODEL_NAME (Duration: ${MODEL_DURATION}s)" >> "$SUMMARY_FILE"
            print_success "Model $MODEL_NAME test succeeded (Duration: ${MODEL_DURATION}s)"
        else
            FAILED_COUNT=$((FAILED_COUNT + 1))
            FAILED_MODELS+=("$MODEL_NAME")
            
            echo "FAILED - $MODEL_NAME" >> "$SUMMARY_FILE"
            print_error "Model $MODEL_NAME test failed"
        fi
        
        echo "" >> "$SUMMARY_FILE"
        
        # If not the last model, clean up and wait
        if [ $CURRENT_NUM -lt $TOTAL_NUM ]; then
            if [ "$CLEANUP_BETWEEN_TESTS" = true ]; then
                echo ""
                print_step "Cleaning up environment, preparing to test next model"
                cleanup_environment
                
                echo ""
                echo "Waiting ${WAIT_BETWEEN_TESTS} seconds before continuing..."
                sleep $WAIT_BETWEEN_TESTS
                
                # Restart User Simulator
                start_user_simulators
            else
                echo ""
                echo "Waiting 10 seconds before continuing..."
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
    echo "End time: $(date)" >> "$SUMMARY_FILE"
    echo "Total duration: ${BATCH_DURATION}s (${BATCH_DURATION_MIN} minutes)" >> "$SUMMARY_FILE"
    echo "Succeeded: $SUCCESS_COUNT" >> "$SUMMARY_FILE"
    echo "Failed: $FAILED_COUNT" >> "$SUMMARY_FILE"
    
    if [ $FAILED_COUNT -gt 0 ]; then
        echo "" >> "$SUMMARY_FILE"
        echo "Failed models:" >> "$SUMMARY_FILE"
        for failed_model in "${FAILED_MODELS[@]}"; do
            echo "  - $failed_model" >> "$SUMMARY_FILE"
        done
    fi
    
    # Display final results
    echo ""
    print_header "Batch Testing Complete"
    
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}Test Summary${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo "  Total models: ${#CHATBOT_MODELS[@]}"
    echo "  Succeeded: $SUCCESS_COUNT"
    echo "  Failed: $FAILED_COUNT"
    echo "  Total duration: ${BATCH_DURATION}s (${BATCH_DURATION_MIN} minutes)"
    echo ""
    echo "  Summary file: $SUMMARY_FILE"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if [ $FAILED_COUNT -gt 0 ]; then
        echo ""
        print_warning "The following models failed testing:"
        for failed_model in "${FAILED_MODELS[@]}"; do
            echo "  - $failed_model"
        done
    fi
    
    echo ""
    
    # Return failure count as exit code
    return $FAILED_COUNT
}

# Stop all services
stop_services() {
    print_step "Stopping all services"
    
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

# Show status
show_status() {
    print_step "Service Status"
    
    echo "User Simulator instances:"
    declare -a GPU_ASSIGNMENTS=("6" "7")
    
    for i in $(seq 0 $((NUM_USER_SIM_INSTANCES - 1))); do
        PORT=$((USER_SIM_BASE_PORT + i))
        INSTANCE_GPU_IDS=${GPU_ASSIGNMENTS[$i]}
        
        if curl -s -f http://localhost:${PORT}/health > /dev/null 2>&1; then
            print_success "Instance $((i + 1)) (port $PORT, GPU $INSTANCE_GPU_IDS) - Running"
        else
            print_error "Instance $((i + 1)) (port $PORT, GPU $INSTANCE_GPU_IDS) - Not running"
        fi
    done
    
    echo ""
    echo "Configured model list:"
    for idx in "${!CHATBOT_MODELS[@]}"; do
        MODEL_CONFIG="${CHATBOT_MODELS[$idx]}"
        MODEL_NAME=$(echo "$MODEL_CONFIG" | cut -d'|' -f1)
        echo "  $((idx + 1)). $MODEL_NAME"
    done
    
    echo ""
    echo "GPU status:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F', ' '{printf "  GPU %s: %s - Utilization %s%% - Memory %s/%s MB\n", $1, $2, $3, $4, $5}'
}

# ==================== Main Menu ====================

# Log path (relative to project root, script should be run from project root: bash for_evaluation/vllm_test_suite.sh)
EVAL_LOG_DIR="./for_evaluation/logs"
EVAL_LOG="$EVAL_LOG_DIR/eval_suite.log"

# Run specified command in background with nohup and tail log
_run_nohup() {
    local cmd="$1"
    mkdir -p "$EVAL_LOG_DIR"
    echo ""
    echo "> Background start: $cmd"
    echo "  Log file: $EVAL_LOG"
    # Use bash -c to ensure child shell correctly expands
    nohup bash -c "cd $(pwd) && $cmd" > "$EVAL_LOG" 2>&1 &
    local PID=$!
    echo "  PID: $PID"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Monitor log:  tail -f $EVAL_LOG"
    echo "  Terminate:    pkill -9 -f vllm_test_suite.sh && pkill -9 -f Baseline_test_local_models_vllm.py && pkill -9 -f vllm_server.py && ray stop --force"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    read -p "Tail log now? [Y/n]: " dotail
    if [[ "$dotail" != "n" && "$dotail" != "N" ]]; then
        tail -f "$EVAL_LOG"
    fi
}

# Kill all related processes
kill_all() {
    print_step "Kill all related processes"
    local SELF_PID=$$

    # Kill other vllm_test_suite.sh processes, excluding self
    local suite_pids
    suite_pids=$(pgrep -f "vllm_test_suite.sh" 2>/dev/null | grep -v "^${SELF_PID}$" || true)
    if [ -n "$suite_pids" ]; then
        echo "$suite_pids" | xargs kill -9 2>/dev/null && echo "  killed: vllm_test_suite.sh (pids: $suite_pids)"
    else
        echo "  (none) vllm_test_suite.sh background processes"
    fi

    pkill -9 -f "Baseline_test_local_models_vllm.py" 2>/dev/null && echo "  killed: Baseline_test_local_models_vllm.py" || echo "  (none) Baseline_test_local_models_vllm.py"
    pkill -9 -f "vllm_server.py" 2>/dev/null && echo "  killed: vllm_server.py" || echo "  (none) vllm_server.py"
    ray stop --force 2>/dev/null || true
    echo "  Ray stopped"
    for port in 5000 5001 5002 5003; do
        local pid
        pid=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$pid" ]; then
            kill -9 $pid 2>/dev/null && echo "  released port $port (pid $pid)"
        fi
    done
    sleep 2
    print_success "Kill complete"
    echo ""
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F', ' '{printf "  GPU %s: %s / %s MB\n", $1, $2, $3}'
}

show_menu() {
    print_header "vLLM Test Suite - Main Menu (Supports Batch Testing)"
    
    echo "Select an action:"
    echo ""
    echo "  1) Batch test all models (recommended)        [background nohup]"
    echo "  2) Full pipeline (clean -> start -> batch)    [background nohup]"
    echo "  3) Clean environment only                     [background nohup]"
    echo "  4) Start User Simulator only                  [background nohup]"
    echo "  5) Stop all services"
    echo "  6) View status"
    echo "  7) View model list"
    echo "  8) Kill all related processes"
    echo "  9) Tail log ($EVAL_LOG)"
    echo "  0) Exit"
    echo ""
    read -p "Enter option [0-9]: " choice
    
    case $choice in
        1)
            _run_nohup "bash for_evaluation/vllm_test_suite.sh batch"
            ;;
        2)
            _run_nohup "bash for_evaluation/vllm_test_suite.sh all"
            ;;
        3)
            _run_nohup "bash for_evaluation/vllm_test_suite.sh cleanup"
            ;;
        4)
            _run_nohup "bash for_evaluation/vllm_test_suite.sh start"
            ;;
        5)
            stop_services
            ;;
        6)
            show_status
            ;;
        7)
            echo ""
            echo "Configured model list (total ${#CHATBOT_MODELS[@]}):"
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
            kill_all
            ;;
        9)
            if [ -f "$EVAL_LOG" ]; then
                tail -f "$EVAL_LOG"
            else
                print_error "Log file does not exist: $EVAL_LOG"
            fi
            ;;
        0)
            echo "Exit"
            exit 0
            ;;
        *)
            print_error "Invalid option"
            exit 1
            ;;
    esac
}

# ==================== Command-Line Argument Processing ====================

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
            echo "Configured model list (total ${#CHATBOT_MODELS[@]}):"
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
            echo "  cleanup/clean  - Clean environment"
            echo "  start          - Start User Simulator"
            echo "  batch/test-all - Batch test all models"
            echo "  stop           - Stop all services"
            echo "  status         - View status"
            echo "  list           - View model list"
            echo "  all/full       - Full pipeline (clean+start+batch test)"
            echo "  help           - Show help"
            echo ""
            echo "No arguments will display interactive menu"
            ;;
        *)
            print_error "Unknown command: $1"
            echo "Run 'bash $0 help' to view help"
            exit 1
            ;;
    esac
fi