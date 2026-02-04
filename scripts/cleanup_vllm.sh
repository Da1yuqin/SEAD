#!/bin/bash
# scripts/cleanup_vllm.sh
# ================================================
# vLLM Service Cleanup Script
# ================================================

echo "🧹 Cleaning up vLLM services..."

# 1. Terminate all vllm_server processes
echo "  - Killing vllm_server processes..."
pkill -9 -f "vllm_server" 2>/dev/null || true
sleep 2

# 2. Terminate all Python processes containing vllm (more thorough)
echo "  - Checking for remaining vllm processes..."
VLLM_PIDS=$(ps aux | grep -i "vllm" | grep -v "grep" | grep -v "cleanup_vllm" | awk '{print $2}')

if [ -n "$VLLM_PIDS" ]; then
    echo "  - Found vllm processes, terminating..."
    echo "$VLLM_PIDS" | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# 3. Clean up Ray temporary directories
echo "  - Cleaning Ray temp directories..."
rm -rf /tmp/ray_* 2>/dev/null || true

# 4. Clean up shared memory
echo "  - Cleaning shared memory..."
rm -rf /dev/shm/ray_* 2>/dev/null || true

# 5. Clean up potentially occupied ports
echo "  - Checking port 5000..."
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null 2>&1 ; then
    PORT_PID=$(lsof -ti:5000 2>/dev/null)
    if [ -n "$PORT_PID" ]; then
        echo "  - Killing process on port 5000 (PID: $PORT_PID)"
        kill -9 $PORT_PID 2>/dev/null || true
        sleep 1
    fi
fi

sleep 2

# 6. Verify cleanup results
REMAINING_PIDS=$(pgrep -f "vllm_server" 2>/dev/null || true)
if [ -n "$REMAINING_PIDS" ]; then
    echo "⚠️  Warning: Some vllm_server processes still running:"
    ps -p $REMAINING_PIDS -o pid,cmd 2>/dev/null || true
    
    # Finally attempt to force kill
    echo "  - Force killing remaining processes..."
    kill -9 $REMAINING_PIDS 2>/dev/null || true
    sleep 2
else
    echo "✅ All vLLM services cleaned up successfully"
fi

# 7. Display GPU status (optional)
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "📊 Current GPU status:"
    GPU_PROCESSES=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | head -5)
    if [ -z "$GPU_PROCESSES" ]; then
        echo "  No GPU processes running"
    else
        echo "$GPU_PROCESSES"
    fi
fi

echo ""