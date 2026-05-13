#!/bin/bash
# scripts/force_cleanup_gpu.sh
# Force cleanup GPU memory

echo "=========================================="
echo "🔴 Force Cleanup GPU Memory"
echo "=========================================="

# 1. Find all GPU processes
echo ""
echo "1️⃣ Finding GPU processes..."
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

# 2. Get PIDs of all GPU processes
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)

if [ -z "$GPU_PIDS" ]; then
    echo "   ℹ️  No GPU processes found"
else
    echo ""
    echo "2️⃣ Found the following GPU processes:"
    for pid in $GPU_PIDS; do
        PROCESS_NAME=$(ps -p $pid -o comm= 2>/dev/null || echo "Exited")
        echo "   PID $pid: $PROCESS_NAME"
    done
    
    echo ""
    echo "3️⃣ Force terminating these processes..."
    for pid in $GPU_PIDS; do
        echo "   Terminating PID $pid..."
        kill -9 $pid 2>/dev/null || true
    done
    
    sleep 2
fi

# 3. Clean up all possible related processes
echo ""
echo "4️⃣ Cleaning up all related processes..."

# Python processes
pkill -9 -f "python.*vllm" 2>/dev/null || true
pkill -9 -f "python.*generate_profiles" 2>/dev/null || true

# vLLM processes
pkill -9 -f "vllm" 2>/dev/null || true

# Ray processes
ray stop --force 2>/dev/null || true
pkill -9 -f "ray" 2>/dev/null || true

# 4. Reset GPU
echo ""
echo "5️⃣ Resetting GPU..."
for i in {0..7}; do
    nvidia-smi -i $i -r 2>/dev/null || true
done

# 5. Clean up shared memory
echo ""
echo "6️⃣ Cleaning up shared memory..."
rm -rf /tmp/ray_* 2>/dev/null || true
rm -rf /dev/shm/ray_* 2>/dev/null || true
rm -rf /tmp/torch_* 2>/dev/null || true
rm -rf /dev/shm/torch_* 2>/dev/null || true

# 6. Wait
echo ""
echo "7️⃣ Waiting for GPU release..."
sleep 5

# 7. Verify
echo ""
echo "8️⃣ Verifying cleanup results:"
echo "=========================================="
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo ""
echo "=========================================="
echo "✅ Cleanup Complete"
echo "=========================================="