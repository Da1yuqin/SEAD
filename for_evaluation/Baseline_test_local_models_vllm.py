#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Local Model Performance Testing Script (vLLM Accelerated) - Using vLLM Engine
Supports user_params filling, correctly handles mutual hangup logic, and supports automatic User vLLM service restart.
"""

# Must be set before importing any CUDA-related libraries
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# Force use of V0 engine (must be before importing vllm)
import os
os.environ['VLLM_USE_V1'] = '0'

import argparse
import sys
import json
import time
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import random
import subprocess
import psutil

# Add project root directory
# Example: Update to your project root path
project_root = "/path/to/your/project"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from verl.utils.reward_score.role_reward import compute_metric_chatbot

# vLLM imports
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    print("⚠️ vLLM not installed, will use transformers (slower)")
    print("   Install command: pip install vllm")
    VLLM_AVAILABLE = False
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

# ==================== UserParamsManager ====================

class UserParamsManager:
    """
    Manages user_params dataset and indexing
    
    Usage Example:
        manager = UserParamsManager(data_dir="./data/user_params")
        batch_params = manager.get_batch_user_params(batch_size=32)
    """
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load datasets
        self.test_user_params = self._load_user_params("test")
        
        # Load indices
        self.test_index = self._load_index("test")
        
        print(f"✅ Loaded {len(self.test_user_params)} test user_params")
        print(f"✅ Test index: {self.test_index}")
        
        # Print first 3 samples
        if self.test_user_params:
            print(f"\n{'='*60}")
            print("Test user_params examples:")
            for i, params in enumerate(self.test_user_params[:3]):
                print(f"  [{i}] {params}")
            print(f"{'='*60}\n")
    
    def _load_user_params(self, split: str) -> list:
        """Load user_params"""
        file_path = self.data_dir / f"{split}_user_params.jsonl"
        if not file_path.exists():
            print(f"⚠️  {file_path} not found, using empty params")
            return []
        
        params_list = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                params_list.append(json.loads(line))
        return params_list
    
    def _load_index(self, split: str) -> int:
        """Load index"""
        index_file = self.data_dir / f"{split}_index.txt"
        if index_file.exists():
            return int(index_file.read_text().strip())
        return 0
    
    def _save_index(self, split: str, index: int):
        """Save index"""
        index_file = self.data_dir / f"{split}_index.txt"
        index_file.write_text(str(index))
    
    def get_batch_user_params(self, batch_size: int) -> list:
        """Get user_params for current batch"""
        params_list = self.test_user_params
        current_index = self.test_index
        split = "test"
        
        if len(params_list) == 0:
            print(f"⚠️  {split} user_params is empty, using empty dicts")
            return [{}] * batch_size
        
        # Check if reset is needed before fetching data
        if current_index >= len(params_list):
            print(f"✅ {split} dataset exhausted (index={current_index}), resetting to 0")
            current_index = 0
            self.test_index = 0
            self._save_index(split, 0)
        
        # Calculate actual available samples
        end_index = min(current_index + batch_size, len(params_list))
        batch_params = params_list[current_index:end_index]
        
        # If insufficient for batch_size, pad with last sample
        while len(batch_params) < batch_size:
            if batch_params:
                batch_params.append(batch_params[-1])
            else:
                batch_params.append({})
        
        # Update index
        new_index = end_index
        self.test_index = new_index
        self._save_index(split, new_index)
        
        return batch_params


def fill_user_params(template: str, user_params: Dict[str, Any]) -> str:
    """
    Fill user_params placeholders.
    Only removes ${xxx} format placeholders, not {xxx}.
    """
    try:
        filled = template
        
        # Field mapping
        field_mapping = {
            'cooperation': 'initial_cooperation',
            'emotion': 'initial_emotion',
            'trust': 'initial_trust',
            'behaviors': 'specific_behaviors',
        }
        
        for key, value in user_params.items():
            # Special handling for behaviors list
            if key == 'behaviors' and isinstance(value, list):
                if value:
                    value_str = '\n'.join([f"- {b}" for b in value])
                else:
                    value_str = "- No specific behavior requirements"
            else:
                value_str = str(value)
            
            # Try multiple placeholder formats
            placeholders_to_try = []
            
            # Original field name
            placeholders_to_try.append(f"${{{key}}}")
            
            # Mapped field name
            mapped_key = field_mapping.get(key)
            if mapped_key:
                placeholders_to_try.append(f"${{{mapped_key}}}")
            
            # Try replacement
            for placeholder in placeholders_to_try:
                if placeholder in filled:
                    filled = filled.replace(placeholder, value_str)
                    break
        
        # Only remove ${xxx} format placeholders, not {xxx}
        filled = re.sub(r'\$\{[^}]+\}', '', filled)
        
        return filled
    except Exception as e:
        print(f"❌ Error filling user_params: {e}")
        return template


# ==================== Global Model Variables ====================
LOCAL_MODEL = None
LOCAL_TOKENIZER = None
USER_SIM_URLS = []
USER_PARAMS_MANAGER = None
DEBUG_PRINT_COUNT = 0

# ==================== vLLM Service Management ====================

def check_vllm_service_health(server_urls: List[str], timeout: int = 5) -> Tuple[List[str], List[str]]:
    """
    Check vLLM service health status
    
    Returns:
        (healthy_urls, unhealthy_urls)
    """
    import requests
    
    healthy = []
    unhealthy = []
    
    for url in server_urls:
        try:
            resp = requests.get(f"{url}/health", timeout=timeout)
            if resp.status_code == 200:
                healthy.append(url)
            else:
                unhealthy.append(url)
        except Exception:
            unhealthy.append(url)
    
    return healthy, unhealthy


def restart_vllm_service(
    restart_script: str,
    model_path: str,
    gpu_ids: str = "0,1",
    mode: str = "call_client",
    max_wait_time: int = 300
) -> bool:
    """
    Restart vLLM service
    
    Args:
        restart_script: Startup script path (e.g., ./scripts/start_vllm.sh)
        model_path: Model path
        gpu_ids: GPU IDs (e.g., "0,1")
        mode: Run mode (default "call_client")
        max_wait_time: Maximum wait time (seconds)
    
    Returns:
        Whether restart succeeded
    """
    print(f"\n{'='*60}")
    print("🔄 User vLLM service anomaly detected, attempting restart...")
    print(f"{'='*60}\n")
    
    # 1. Kill old processes
    print("1️⃣ Cleaning up old processes...")
    try:
        # Find and kill vllm-related processes
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'vllm' in cmdline.lower() and 'user' in cmdline.lower():
                    print(f"   Terminating process: PID={proc.info['pid']}")
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        time.sleep(5)  # Wait for processes to fully exit
        print("   ✅ Old process cleanup complete\n")
    except Exception as e:
        print(f"   ⚠️ Error during process cleanup: {e}\n")
    
    # 2. Start new service
    print("2️⃣ Starting new service...")
    try:
        # Build startup command
        cmd = [
            "bash",
            restart_script,
            model_path,
            "1",  # tensor_parallel_size
            "1",  # num_instances
            mode
        ]
        
        # Set environment variables
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu_ids
        
        print(f"   Executing command: {' '.join(cmd)}")
        print(f"   GPU IDs: {gpu_ids}\n")
        
        # Start in background
        subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True
        )
        
        print("   ✅ Startup command sent\n")
    except Exception as e:
        print(f"   ❌ Startup failed: {e}\n")
        return False
    
    # 3. Wait for service to be ready
    print("3️⃣ Waiting for service to be ready...")
    start_time = time.time()
    
    # Extract host and port from URL
    import requests
    from urllib.parse import urlparse
    
    # Assume first URL is main service
    test_url = "http://localhost:5000"  # Default value
    if USER_SIM_URLS:
        test_url = USER_SIM_URLS[0]
    
    while time.time() - start_time < max_wait_time:
        try:
            resp = requests.get(f"{test_url}/health", timeout=5)
            if resp.status_code == 200:
                elapsed = time.time() - start_time
                print(f"   ✅ Service ready (took {elapsed:.1f}s)\n")
                return True
        except Exception:
            pass
        
        # Print progress every 10 seconds
        elapsed = time.time() - start_time
        if int(elapsed) % 10 == 0:
            print(f"   ⏳ Waiting... ({elapsed:.0f}s / {max_wait_time}s)")
        
        time.sleep(2)
    
    print(f"   ❌ Service startup timeout (>{max_wait_time}s)\n")
    return False


def ensure_vllm_service_available(
    server_urls: List[str],
    restart_script: str,
    model_path: str,
    gpu_ids: str = "0,1",
    max_retries: int = 3
) -> List[str]:
    """
    Ensure vLLM service is available, automatically restart if necessary
    
    Returns:
        List of available service URLs
    """
    for attempt in range(max_retries):
        healthy, unhealthy = check_vllm_service_health(server_urls)
        
        if healthy:
            if unhealthy:
                print(f"⚠️ Some services abnormal: {len(unhealthy)}/{len(server_urls)}")
            return healthy
        
        # All services down, attempt restart
        print(f"\n❌ All User vLLM services unavailable (attempt {attempt+1}/{max_retries})")
        
        if attempt < max_retries - 1:
            success = restart_vllm_service(
                restart_script=restart_script,
                model_path=model_path,
                gpu_ids=gpu_ids
            )
            
            if not success:
                print(f"⚠️ Restart failed, waiting 30 seconds before retry...\n")
                time.sleep(30)
        else:
            print(f"\n❌ Maximum retries reached, unable to recover service\n")
            raise RuntimeError("User vLLM service cannot be started")
    
    return []


# ==================== Model Loading Functions (vLLM Version) ====================

def load_local_model_vllm(
    model_path: str,
    tensor_parallel_size: int = 5,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 8192
):
    """
    Load local model using vLLM (V0 engine + Ray backend)
    
    Example:
        load_local_model_vllm(
            model_path="/path/to/your/model",
            tensor_parallel_size=4,
            gpu_memory_utilization=0.9
        )
    """
    global LOCAL_MODEL, LOCAL_TOKENIZER
    
    print(f"\n{'='*80}")
    print(f"Loading local model with vLLM: {model_path}")
    print(f"{'='*80}\n")
    
    # Check path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    # Verify environment variables
    v1_status = os.environ.get('VLLM_USE_V1', '1')
    print(f"Environment check:")
    print(f"  - VLLM_USE_V1: {v1_status} {'✅ (V0 engine)' if v1_status == '0' else '⚠️ (V1 engine)'}")
    print()
    
    print(f"vLLM configuration:")
    print(f"  - Engine: V0")
    print(f"  - Distributed Backend: Ray (avoids CUDA fork issues)")
    print(f"  - Tensor Parallel Size: {tensor_parallel_size}")
    print(f"  - GPU Memory Utilization: {gpu_memory_utilization}")
    print(f"  - Max Model Length: {max_model_len}")
    print(f"  - Data Type: bfloat16")
    print()
    
    print("Loading vLLM model (this may take a few minutes)...")
    
    try:
        LOCAL_MODEL = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=True,
            disable_log_stats=True,
            distributed_executor_backend='ray',
        )
        
        # Get tokenizer
        LOCAL_TOKENIZER = LOCAL_MODEL.get_tokenizer()
        
        print(f"✓ vLLM model loaded successfully!")
        print(f"  - Tokenizer: {type(LOCAL_TOKENIZER).__name__}")
        print()
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def load_local_model_transformers(model_path: str):
    """
    Load local model using transformers (fallback option)
    
    Example:
        load_local_model_transformers(model_path="/path/to/your/model")
    """
    global LOCAL_MODEL, LOCAL_TOKENIZER
    
    print(f"\n{'='*80}")
    print(f"Loading local model with Transformers: {model_path}")
    print(f"{'='*80}\n")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    
    print("Loading tokenizer...")
    LOCAL_TOKENIZER = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side='left'
    )
    
    if LOCAL_TOKENIZER.pad_token is None:
        LOCAL_TOKENIZER.pad_token = LOCAL_TOKENIZER.eos_token
    
    print("Loading model...")
    LOCAL_MODEL = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    LOCAL_MODEL.eval()
    
    print(f"✓ Transformers model loaded successfully!")
    print(f"  - Parameters: {sum(p.numel() for p in LOCAL_MODEL.parameters()) / 1e9:.2f}B")
    print()


# ==================== Helper Functions ====================

def extract_tag_value(text: str, tag_name: str, cast_type=None):
    """Extract value from XML-like tags"""
    if not text:
        return None
    pattern = f'<{tag_name}[^>]*>(.*?)</{tag_name}>'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        value_str = match.group(1).strip()
        if cast_type:
            try:
                return cast_type(value_str)
            except (ValueError, TypeError):
                return None
        return value_str
    return None


# ==================== Core Calling Functions (vLLM Version) ====================

def call_local_model_batch_vllm(
    batch_messages: List[List[Dict]], 
    temperature: float = 0.7, 
    max_new_tokens: int = 512
) -> List[Dict]:
    """
    Batch inference using vLLM (high-performance version)
    Supports user_params filling
    """
    global LOCAL_MODEL, LOCAL_TOKENIZER, USER_PARAMS_MANAGER, DEBUG_PRINT_COUNT
    
    if LOCAL_MODEL is None or LOCAL_TOKENIZER is None:
        return [
            {'content': '', 'latency': 0, 'tokens': 0, 'cost': 0, 'status': 'failed', 'error': 'Model not loaded'}
            for _ in batch_messages
        ]
    
    batch_size = len(batch_messages)
    start_time = time.time()
    
    try:
        # 1. Get user_params
        if USER_PARAMS_MANAGER is not None:
            batch_user_params = USER_PARAMS_MANAGER.get_batch_user_params(batch_size)
        else:
            batch_user_params = [{}] * batch_size
        
        # 2. Fill user_params and apply chat template
        prompts = []
        for messages, user_params in zip(batch_messages, batch_user_params):
            # Fill placeholders in system prompt
            if messages and messages[0]['role'] == 'system':
                original_system = messages[0]['content']
                filled_system = fill_user_params(original_system, user_params)
                messages[0]['content'] = filled_system
            
            # Apply chat template
            prompt = LOCAL_TOKENIZER.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)
        
        # Debug: Print first 3 batches, then 5% probability
        if DEBUG_PRINT_COUNT < 3 or random.random() < 0.05:
            DEBUG_PRINT_COUNT += 1
            print(f"\n{'='*60}")
            print(f"【DEBUG #{DEBUG_PRINT_COUNT}】Filled Prompt Example:")
            print(f"User Params: {batch_user_params[0]}")
            print(f"Prompt (first 1000 chars):\n{prompts[0][:1000]}")
            print(f"{'='*60}\n")
        
        # 3. Set sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.9,
            max_tokens=max_new_tokens,
            skip_special_tokens=True,
        )
        
        # 4. vLLM batch generation
        outputs = LOCAL_MODEL.generate(prompts, sampling_params)
        
        # 5. Process results
        total_latency = time.time() - start_time
        avg_latency = total_latency / batch_size
        
        results = []
        for output in outputs:
            response = output.outputs[0].text
            num_tokens = len(output.outputs[0].token_ids)
            
            results.append({
                'content': response,
                'latency': avg_latency,
                'tokens': num_tokens,
                'cost': 0.0,
                'status': 'success'
            })
        
        print(f"    ✓ vLLM batch inference: {batch_size} samples, total time {total_latency:.2f}s, avg {avg_latency:.2f}s/sample")
        
        return results
        
    except Exception as e:
        print(f"    ✗ vLLM inference failed: {e}")
        latency = time.time() - start_time
        return [
            {
                'content': '',
                'latency': latency / batch_size,
                'tokens': 0,
                'cost': 0.0,
                'status': 'failed',
                'error': str(e)
            }
            for _ in batch_messages
        ]


def call_local_model_batch_transformers(
    batch_messages: List[List[Dict]], 
    temperature: float = 0.7, 
    max_new_tokens: int = 512
) -> List[Dict]:
    """
    Batch inference using transformers (fallback option)
    Supports user_params filling
    """
    global LOCAL_MODEL, LOCAL_TOKENIZER, USER_PARAMS_MANAGER, DEBUG_PRINT_COUNT
    
    if LOCAL_MODEL is None or LOCAL_TOKENIZER is None:
        return [
            {'content': '', 'latency': 0, 'tokens': 0, 'cost': 0, 'status': 'failed', 'error': 'Model not loaded'}
            for _ in batch_messages
        ]
    
    batch_size = len(batch_messages)
    start_time = time.time()
    
    try:
        # 1. Get user_params
        if USER_PARAMS_MANAGER is not None:
            batch_user_params = USER_PARAMS_MANAGER.get_batch_user_params(batch_size)
        else:
            batch_user_params = [{}] * batch_size
        
        # 2. Fill user_params and apply chat template
        all_texts = []
        for messages, user_params in zip(batch_messages, batch_user_params):
            # Fill system prompt
            if messages and messages[0]['role'] == 'system':
                original_system = messages[0]['content']
                filled_system = fill_user_params(original_system, user_params)
                messages[0]['content'] = filled_system
            
            text = LOCAL_TOKENIZER.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            all_texts.append(text)
        
        # Debug print
        if DEBUG_PRINT_COUNT < 3 or random.random() < 0.05:
            DEBUG_PRINT_COUNT += 1
            print(f"\n{'='*60}")
            print(f"【DEBUG #{DEBUG_PRINT_COUNT}】Filled Prompt Example:")
            print(f"User Params: {batch_user_params[0]}")
            print(f"Prompt (first 1000 chars):\n{all_texts[0][:1000]}")
            print(f"{'='*60}\n")
        
        # Tokenize
        inputs = LOCAL_TOKENIZER(
            all_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=8192
        ).to(LOCAL_MODEL.device)
        
        input_lengths = inputs['attention_mask'].sum(dim=1).tolist()
        
        # Generate
        with torch.no_grad():
            outputs = LOCAL_MODEL.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                pad_token_id=LOCAL_TOKENIZER.pad_token_id,
                eos_token_id=LOCAL_TOKENIZER.eos_token_id,
            )
        
        # Decode
        total_latency = time.time() - start_time
        avg_latency = total_latency / batch_size
        
        results = []
        for i, (output, input_len) in enumerate(zip(outputs, input_lengths)):
            generated_tokens = output[input_len:]
            response = LOCAL_TOKENIZER.decode(generated_tokens, skip_special_tokens=True)
            
            results.append({
                'content': response,
                'latency': avg_latency,
                'tokens': len(output),
                'cost': 0.0,
                'status': 'success'
            })
        
        print(f"    ✓ Transformers batch inference: {batch_size} samples, total time {total_latency:.2f}s, avg {avg_latency:.2f}s/sample")
        
        return results
        
    except Exception as e:
        print(f"    ✗ Transformers inference failed: {e}")
        latency = time.time() - start_time
        return [
            {
                'content': '',
                'latency': latency / batch_size,
                'tokens': 0,
                'cost': 0.0,
                'status': 'failed',
                'error': str(e)
            }
            for _ in batch_messages
        ]


# Automatically select inference function
def call_local_model_batch(batch_messages, temperature=0.7, max_new_tokens=512):
    """Automatically select vLLM or transformers"""
    if VLLM_AVAILABLE and isinstance(LOCAL_MODEL, LLM):
        return call_local_model_batch_vllm(batch_messages, temperature, max_new_tokens)
    else:
        return call_local_model_batch_transformers(batch_messages, temperature, max_new_tokens)


# ==================== User Simulator Calling (Multi-instance Load Balancing + Auto Restart) ====================

def call_batch_user_simulator(
    batch_dialogue_histories: List[List[Dict]],
    batch_turn_ids: List[int],
    batch_statuses_list: List[List[int]],
    server_urls: List[str] = None,
    timeout: int = 300,
    restart_config: Dict = None
) -> List[Dict]:
    """
    Batch call User Simulator (supports multi-instance load balancing + auto restart)
    
    Example:
        results = call_batch_user_simulator(
            batch_dialogue_histories=histories,
            batch_turn_ids=turn_ids,
            batch_statuses_list=statuses,
            server_urls=["http://localhost:5000", "http://localhost:5001"],
            restart_config={
                'restart_script': './scripts/start_vllm.sh',
                'model_path': '/path/to/model',
                'gpu_ids': '0,1'
            }
        )
    """
    import requests
    
    global USER_SIM_URLS
    
    if server_urls is None:
        server_urls = USER_SIM_URLS
    
    if not server_urls:
        server_urls = ["http://localhost:5000"]
    
    # Check service health before calling
    if restart_config:
        try:
            available_urls = ensure_vllm_service_available(
                server_urls=server_urls,
                restart_script=restart_config['restart_script'],
                model_path=restart_config['model_path'],
                gpu_ids=restart_config.get('gpu_ids', '0,1'),
                max_retries=restart_config.get('max_retries', 3)
            )
            server_urls = available_urls
        except Exception as e:
            print(f"❌ Service recovery failed: {e}")
            # Return failure results
            batch_size = len(batch_dialogue_histories)
            default_fail = {
                'content': '',
                'clean_content': '',
                'latency': 0,
                'status': 'error',
                'status_value': 0,
                'error': 'Service unavailable'
            }
            return [default_fail] * batch_size
    
    # Randomly select a User Sim instance (load balancing)
    server_url = random.choice(server_urls)
    
    batch_payload = {
        'batch_dialogue_histories': batch_dialogue_histories,
        'batch_turn_ids': batch_turn_ids,
        'batch_statuses_list': batch_statuses_list,
        'mode': 'call_client'
    }
    
    start_time = time.time()
    batch_size = len(batch_dialogue_histories)
    default_fail = {
        'content': '',
        'clean_content': '',
        'latency': 0,
        'status': 'error',
        'status_value': 0,
        'error': 'Batch call failed'
    }
    
    try:
        resp = requests.post(f'{server_url}/dialogue_batch', json=batch_payload, timeout=timeout)
        latency = (time.time() - start_time) / batch_size
        
        if resp.status_code == 200:
            response_data = resp.json()
            raw_results = response_data.get('batch_results', [])
            
            processed_results = []
            for res in raw_results:
                raw_response = res.get('raw_response', res.get('response', ''))
                
                user_reply_match = re.search(r'<user_reply[^>]*>(.*?)</user_reply>', raw_response, re.DOTALL | re.IGNORECASE)
                user_think_match = re.search(r'<user_think[^>]*>(.*?)</user_think>', raw_response, re.DOTALL | re.IGNORECASE)
                
                user_reply_text = user_reply_match.group(1).strip() if user_reply_match else ""
                user_think_text = user_think_match.group(1).strip() if user_think_match else ""
                
                status_value = res.get('status_value')
                if status_value is None:
                    status_match = re.search(r'<user_status[^>]*>(.*?)</user_status>', raw_response, re.DOTALL | re.IGNORECASE)
                    if status_match:
                        try:
                            status_value = int(status_match.group(1).strip())
                        except:
                            status_value = 0
                    else:
                        status_value = 0
                else:
                    status_value = int(status_value)
                
                # Save complete raw_response (including all tags)
                full_content = raw_response if raw_response else ""
                if not full_content:
                    full_content = ""
                    if user_think_text: full_content += f"<user_think>{user_think_text}</user_think>\n"
                    if user_reply_text: full_content += f"<user_reply>{user_reply_text}</user_reply>\n"
                    full_content += f"<user_status>{status_value}</user_status>"
                
                clean_content = ""
                if user_think_text: clean_content += f"<user_think>{user_think_text}</user_think>\n"
                if user_reply_text: clean_content += f"<user_reply>{user_reply_text}</user_reply>"
                
                processed_results.append({
                    'content': full_content,
                    'clean_content': clean_content,
                    'latency': latency,
                    'status': 'success',
                    'status_value': status_value,
                })
            
            while len(processed_results) < batch_size:
                processed_results.append(default_fail)
            return processed_results
        else:
            print(f"  ⚠️ User Sim Error: {resp.status_code} (URL: {server_url})")
            return [default_fail] * batch_size
    
    except requests.exceptions.Timeout:
        print(f"  ⚠️ User Sim Timeout: {timeout}s (URL: {server_url})")
        return [default_fail] * batch_size
    except requests.exceptions.ConnectionError:
        print(f"  ⚠️ User Sim Connection Error (URL: {server_url})")
        return [default_fail] * batch_size
    except Exception as e:
        print(f"  ⚠️ User Sim Exception: {e} (URL: {server_url})")
        return [default_fail] * batch_size


# ==================== Batch Processing Logic ====================

class ConversationState:
    """Manages state for a single conversation"""
    def __init__(self, idx, initial_prompt):
        self.idx = idx
        self.initial_prompt = initial_prompt
        self.dialogue_history = []
        self.ground_truth_history = []
        self.actor_history = []
        self.chatbot_messages = [{"role": "user", "content": initial_prompt}]
        self.batch_statuses = []
        self.turn_id = 0
        self.is_finished = False
        self.chatbot_wants_end = False
        self.chatbot_status = 0
        self.stats = {
            'chatbot_latency': [],
            'user_latency': [],
            'chatbot_tokens': [],
            'chatbot_cost': 0.0,
            'total_turns': 0,
            'final_status': 0
        }

def process_batch(
    batch_data, 
    user_sim_urls, 
    f_log, 
    max_turns=16, 
    timeout_turns=15,
    restart_config=None
):
    """
    Process a batch of conversations.
    Correctly sets final_status to distinguish success, misjudgment, failure, and timeout.
    Supports automatic User vLLM service restart.
    """
    all_conversations = [ConversationState(idx, prompt) for idx, prompt in batch_data]
    active_conversations = all_conversations.copy()
    
    print(f"  > Starting batch processing, size: {len(all_conversations)}")
    
    for turn in range(max_turns):
        if not active_conversations:
            break
        
        print(f"\n  === Turn {turn+1}/{max_turns} (Active conversations: {len(active_conversations)}) ===")
        
        # Batch call Chatbot
        batch_messages = [conv.chatbot_messages for conv in active_conversations]
        batch_results = call_local_model_batch(batch_messages)
        
        # Process Chatbot responses
        for conv, result in zip(active_conversations, batch_results):
            if result['status'] != 'success':
                conv.is_finished = True
                conv.stats['final_status'] = -1
                conv.stats['total_turns'] = turn + 1
                continue
            
            chatbot_response = result['content']
            conv.stats['chatbot_latency'].append(result['latency'])
            conv.stats['chatbot_tokens'].append(result['tokens'])
            conv.stats['chatbot_cost'] += result['cost']
            
            # Save complete Chatbot output
            conv.ground_truth_history.append({'role': 'assistant', 'content': chatbot_response})
            
            # Save simplified version
            response_match = re.search(r'<response[^>]*>(.*?)</response>', chatbot_response, re.DOTALL | re.IGNORECASE)
            clean_response = response_match.group(1).strip() if response_match else re.sub(r'<[^>]+>', '', chatbot_response).strip()
            conv.dialogue_history.append({'role': 'assistant', 'content': f"<response>{clean_response}</response>"})
            conv.actor_history.append({'role': 'assistant', 'content': f"<response>{clean_response}</response>"})
            
            # Extract Chatbot's status
            predicted_status = extract_tag_value(chatbot_response, 'status', int)
            if predicted_status is not None:
                conv.chatbot_status = predicted_status
                conv.chatbot_wants_end = (predicted_status == 1)
            else:
                conv.chatbot_status = 0
                conv.chatbot_wants_end = False
            
            # If Chatbot proactively hangs up (status=1), end conversation immediately
            if conv.chatbot_status == 1:
                conv.stats['total_turns'] = turn + 1
                conv.is_finished = True
                
                # Determine final_status based on previous round's user_status
                if conv.batch_statuses:
                    last_user_status = conv.batch_statuses[-1]
                    if last_user_status == 1:
                        # Scenario 2: User agreed, Chatbot hangs up → success
                        conv.stats['final_status'] = 1
                        print(f"    [Sample {conv.idx}] [Turn {turn+1}] ✅ Chatbot correctly hung up (user agreed, success)")
                    else:
                        # Scenario 1: User hasn't agreed, Chatbot hangs up → misjudgment (recorded as timeout)
                        conv.stats['final_status'] = 0
                        print(f"    [Sample {conv.idx}] [Turn {turn+1}] ☎️ Chatbot hung up (user not agreed, misjudgment)")
                else:
                    # Scenario 3: Hung up on first turn → failure
                    conv.stats['final_status'] = -1
                    print(f"    [Sample {conv.idx}] [Turn {turn+1}] ❌ Chatbot hung up on first turn (failure)")
                
                continue

        # Filter: Only call User Simulator for unfinished conversations
        sim_candidates = [c for c in active_conversations if not c.is_finished]
        if not sim_candidates:
            break

        # Batch call User Simulator
        batch_histories = [c.actor_history for c in sim_candidates]
        batch_turn_ids = [c.turn_id for c in sim_candidates]
        batch_statuses_lists = [c.batch_statuses for c in sim_candidates]
        
        print(f"  > Calling User Simulator: {len(sim_candidates)} samples (using {len(user_sim_urls)} instances)")
        sim_results = call_batch_user_simulator(
            batch_histories, 
            batch_turn_ids, 
            batch_statuses_lists, 
            server_urls=user_sim_urls,
            restart_config=restart_config
        )
        
        # Process User responses
        for i, conv in enumerate(sim_candidates):
            res = sim_results[i]
            if res['status'] != 'success':
                conv.is_finished = True
                conv.stats['final_status'] = -1
                conv.stats['total_turns'] = turn + 1
                continue
            
            conv.stats['user_latency'].append(res['latency'])
            
            # Save complete User output
            conv.ground_truth_history.append({'role': 'user', 'content': res['content']})
            
            # Save simplified version
            conv.dialogue_history.append({'role': 'user', 'content': res['clean_content']})
            conv.actor_history.append({'role': 'user', 'content': res['clean_content']})
            
            conv.turn_id += 1
            user_status = res['status_value']
            conv.batch_statuses.append(user_status)
            
            # Only determine end based on user_status
            # 1. User explicitly agrees
            if user_status == 1:
                conv.stats['final_status'] = 1
                conv.stats['total_turns'] = turn + 1
                conv.is_finished = True
                print(f"    [Sample {conv.idx}] [Turn {turn+1}] ✅ User agreed (success)")
            
            # 2. User explicitly hangs up
            elif user_status == -1:
                conv.stats['final_status'] = -1
                conv.stats['total_turns'] = turn + 1
                conv.is_finished = True
                print(f"    [Sample {conv.idx}] [Turn {turn+1}] ❌ User hung up (failure)")
            
            # 3. Timeout check
            elif turn + 1 >= timeout_turns:
                conv.stats['final_status'] = 0
                conv.stats['total_turns'] = turn + 1
                conv.is_finished = True
                print(f"    [Sample {conv.idx}] [Turn {turn+1}] ⏱️ Conversation timeout (incomplete)")
            
            # Update Chatbot message history
            if not conv.is_finished:
                conv.chatbot_messages.append({'role': 'assistant', 'content': conv.ground_truth_history[-2]['content']})
                conv.chatbot_messages.append({'role': 'user', 'content': res['clean_content']})
        
        # Update active conversation list
        active_conversations = [c for c in active_conversations if not c.is_finished]

    # Handle unfinished conversations (timeout)
    for conv in active_conversations:
        conv.stats['final_status'] = 0
        conv.stats['total_turns'] = max_turns
        conv.is_finished = True

    # Write to log
    for conv in all_conversations:
        if conv.stats['total_turns'] == 0:
            conv.stats['total_turns'] = len(conv.dialogue_history) // 2
        log_entry = {
            "sample_id": conv.idx,
            "prompt": conv.initial_prompt,
            "dialogue_history": conv.dialogue_history,
            "ground_truth_history": conv.ground_truth_history,
            "final_status": conv.stats['final_status'],
            "total_turns": conv.stats['total_turns'],
            "chatbot_wants_end": conv.chatbot_wants_end,
            "chatbot_status": conv.chatbot_status,
            "timestamp": datetime.now().isoformat()
        }
        f_log.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        f_log.flush()

    return [c.ground_truth_history for c in all_conversations], [c.stats for c in all_conversations]

# ==================== Main Testing Logic ====================

def test_model_on_dataset(
    model_name: str,
    test_data: pd.DataFrame,
    output_dir: Path,
    user_sim_urls: List[str],
    batch_size: int = 64,
    max_samples: int = None,
    restart_config: Dict = None
) -> Dict:
    """
    Test model on dataset
    
    Example:
        result = test_model_on_dataset(
            model_name="my-model",
            test_data=test_df,
            output_dir=Path("./outputs"),
            user_sim_urls=["http://localhost:5000"],
            batch_size=32,
            restart_config={
                'restart_script': './scripts/start_vllm.sh',
                'model_path': '/path/to/model',
                'gpu_ids': '0,1'
            }
        )
    """
    print(f"\n{'='*80}")
    print(f"Testing local model: {model_name} | Batch Size: {batch_size}")
    print(f"Using engine: {'vLLM' if VLLM_AVAILABLE else 'Transformers'}")
    print(f"User Sim instances: {len(user_sim_urls)}")
    
    if restart_config:
        print(f"✅ Auto-restart feature: Enabled")
    else:
        print(f"⚠️ Auto-restart feature: Disabled")
    
    print(f"{'='*80}\n")
    
    # Check User Simulator service
    print("Checking User Simulator service...")
    import requests
    available_urls = []
    for url in user_sim_urls:
        try:
            resp = requests.get(f"{url}/health", timeout=5)
            if resp.status_code == 200:
                print(f"  ✅ {url} - Normal")
                available_urls.append(url)
            else:
                print(f"  ⚠️ {url} - Abnormal (status code: {resp.status_code})")
        except Exception as e:
            print(f"  ❌ {url} - Cannot connect ({e})")
    
    if not available_urls:
        if restart_config:
            print("\n⚠️ No available User Simulator services, attempting auto-restart...")
            try:
                available_urls = ensure_vllm_service_available(
                    server_urls=user_sim_urls,
                    restart_script=restart_config['restart_script'],
                    model_path=restart_config['model_path'],
                    gpu_ids=restart_config.get('gpu_ids', '0,1'),
                    max_retries=restart_config.get('max_retries', 3)
                )
            except Exception as e:
                print(f"\n❌ Service restart failed: {e}")
                return None
        else:
            print("\n❌ No available User Simulator services!")
            return None
    
    print(f"\nAvailable User Sim instances: {len(available_urls)}/{len(user_sim_urls)}\n")
    
    if max_samples:
        test_data = test_data.head(max_samples)
    
    n_samples = len(test_data)
    print(f"Total test samples: {n_samples}")
    
    all_ground_truth_histories = []
    all_stats = []
    
    dialogue_log_file = output_dir / f"{model_name}_dialogues.jsonl"
    print(f"Dialogue logs will be saved to: {dialogue_log_file}\n")
    
    data_list = []
    for idx, row in test_data.iterrows():
        data_list.append((idx, row['prompt'][0]['content']))
    
    # Monitoring variables
    processed_count = 0
    success_count = 0
    monitor_interval = 5
    
    overall_start_time = time.time()
    
    with open(dialogue_log_file, 'w', encoding='utf-8') as f_log:
        for i in range(0, n_samples, batch_size):
            batch_slice = data_list[i : i + batch_size]
            print(f"\n{'='*60}")
            print(f"Processing Batch {i//batch_size + 1} / {(n_samples + batch_size - 1)//batch_size}")
            print(f"{'='*60}")
            
            b_ground_truth_histories, b_stats = process_batch(
                batch_slice, 
                available_urls, 
                f_log,
                restart_config=restart_config
            )
            
            all_ground_truth_histories.extend(b_ground_truth_histories)
            all_stats.extend(b_stats)
            
            # Real-time monitoring
            for s in b_stats:
                processed_count += 1
                if s['final_status'] == 1:
                    success_count += 1
            
            if processed_count % monitor_interval == 0 or processed_count == n_samples:
                curr_rate = (success_count / processed_count) * 100
                elapsed_time = time.time() - overall_start_time
                avg_time_per_sample = elapsed_time / processed_count
                eta = avg_time_per_sample * (n_samples - processed_count)
                
                print(f"\n{'*'*60}")
                print(f"📊 Real-time Monitoring (Processed: {processed_count}/{n_samples})")
                print(f"   Current successes: {success_count}")
                print(f"   Current completion rate: {curr_rate:.2f}%")
                print(f"   Elapsed time: {elapsed_time/60:.1f} minutes")
                print(f"   Estimated remaining: {eta/60:.1f} minutes")
                print(f"{'*'*60}\n")
    
    total_time = time.time() - overall_start_time
    print(f"\n✓ All batches completed! Total time: {total_time/60:.1f} minutes\n")
    
    # Calculate evaluation metrics
    print("Calculating evaluation metrics...")
    metrics = {}
    try:
        metrics_result = compute_metric_chatbot(
            ground_truth_histories=all_ground_truth_histories,
            current_step=0,
            meta_info={'model': model_name}
        )
        metrics = metrics_result['metrics']
    except Exception as e:
        print(f"⚠️  Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
    
    performance_stats = {
        'avg_chatbot_latency': float(np.mean([np.mean(s['chatbot_latency']) if s['chatbot_latency'] else 0 for s in all_stats])),
        'avg_user_latency': float(np.mean([np.mean(s['user_latency']) if s['user_latency'] else 0 for s in all_stats])),
        'avg_tokens_per_turn': float(np.mean([np.mean(s['chatbot_tokens']) if s['chatbot_tokens'] else 0 for s in all_stats])),
        'total_tokens': sum([sum(s['chatbot_tokens']) for s in all_stats]),
        'total_cost': 0.0,
        'avg_cost_per_dialogue': 0.0,
        'avg_turns': float(np.mean([s['total_turns'] for s in all_stats])),
        'total_time_minutes': total_time / 60,
    }
    
    total_latency = sum([sum(s['chatbot_latency']) for s in all_stats])
    performance_stats['tokens_per_second'] = performance_stats['total_tokens'] / total_latency if total_latency > 0 else 0
    
    result = {
        'model': model_name,
        'n_samples': n_samples,
        'task_metrics': metrics,
        'performance_stats': performance_stats,
        'timestamp': datetime.now().isoformat(),
        'engine': 'vLLM' if VLLM_AVAILABLE else 'Transformers',
        'num_user_sim_instances': len(available_urls)
    }
    
    output_file = output_dir / f"{model_name}_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Summary results saved to: {output_file}\n")
    
    print(f"{'='*80}")
    print(f"Model {model_name} Test Results Summary")
    print(f"{'='*80}\n")
    
    if metrics:
        print(f"【Task Metrics】")
        print(f"  Completion Rate:     {metrics.get('completion_rate', 0):.2%}")
        print(f"  False Positive Rate: {metrics.get('false_positive_rate', 0):.2%}")
        print(f"  Avg Turns to Target: {metrics.get('average_turns_to_target', 0):.2f}")
    
    print(f"\n【Performance Metrics】")
    print(f"  Inference Engine:    {'vLLM' if VLLM_AVAILABLE else 'Transformers'}")
    print(f"  User Sim Instances:  {len(available_urls)}")
    print(f"  Average Latency:     {performance_stats['avg_chatbot_latency']:.3f} s/turn")
    print(f"  Throughput:          {performance_stats['tokens_per_second']:.1f} tokens/s")
    print(f"  Avg Tokens:          {performance_stats['avg_tokens_per_turn']:.1f} tokens/turn")
    print(f"  Total Time:          {performance_stats['total_time_minutes']:.1f} minutes")
    print(f"\n【Cost Statistics】")
    print(f"  Total Cost:          ¥0.00 (local model)")
    print(f"\n{'='*80}\n")
    
    return result


# ==================== Main Function ====================

def main():
    parser = argparse.ArgumentParser(description="Test local model performance (vLLM accelerated + auto-restart)")
    parser.add_argument("--model_path", required=True, help="Local model path")
    parser.add_argument("--model_name", default="qwen-local", help="Model name")
    parser.add_argument("--test_data", default="./outputs/evaluation/test_set/test_chatbot.parquet", help="Test data path")
    parser.add_argument("--n_samples", type=int, default=None, help="Number of test samples")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--output_dir", default="./outputs/evaluation", help="Output directory")
    parser.add_argument("--user_sim_urls", nargs='+', default=["http://localhost:5000"], help="User Simulator URLs")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Tensor parallel size")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization")
    parser.add_argument("--use_transformers", action='store_true', help="Force use of transformers")
    parser.add_argument("--user_params_dir", type=str, default="./outputs/evaluation/test_set/user_param", help="User params directory")
    
    # vLLM service restart parameters
    parser.add_argument("--vllm_restart_script", type=str, 
                        default="./scripts/start_vllm.sh",
                        help="User vLLM service startup script path")
    parser.add_argument("--vllm_model_path", type=str,
                        help="User vLLM model path (for restart)")
    parser.add_argument("--vllm_gpu_ids", type=str, default="0,1",
                        help="User vLLM GPU IDs")
    parser.add_argument("--disable_auto_restart", action='store_true',
                        help="Disable auto-restart feature")
    
    args = parser.parse_args()
    
    global USER_SIM_URLS, USER_PARAMS_MANAGER
    USER_SIM_URLS = args.user_sim_urls
    
    # Initialize UserParamsManager
    print(f"\n{'='*80}")
    print("Initializing User Params Manager")
    print(f"{'='*80}\n")
    try:
        USER_PARAMS_MANAGER = UserParamsManager(args.user_params_dir)
    except Exception as e:
        print(f"⚠️  Unable to load user_params: {e}")
        print("   Will use empty user_params")
        USER_PARAMS_MANAGER = None
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build restart configuration
    restart_config = None
    if not args.disable_auto_restart and args.vllm_model_path:
        restart_config = {
            'restart_script': args.vllm_restart_script,
            'model_path': args.vllm_model_path,
            'gpu_ids': args.vllm_gpu_ids,
            'max_retries': 3
        }
        print(f"\n{'='*80}")
        print("✅ Auto-restart feature enabled")
        print(f"{'='*80}")
        print(f"   Script: {restart_config['restart_script']}")
        print(f"   Model: {restart_config['model_path']}")
        print(f"   GPU: {restart_config['gpu_ids']}")
        print(f"   Max retries: {restart_config['max_retries']}\n")
    else:
        print(f"\n{'='*80}")
        print("⚠️ Auto-restart feature disabled")
        print(f"{'='*80}\n")
        if not args.vllm_model_path:
            print("   Reason: --vllm_model_path parameter not provided\n")
    
    # Load model
    try:
        if VLLM_AVAILABLE and not args.use_transformers:
            load_local_model_vllm(
                args.model_path,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization
            )
        else:
            if args.use_transformers:
                print("⚠️ Using --use_transformers parameter, will use Transformers engine")
            load_local_model_transformers(args.model_path)
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Read test data
    print(f"\nReading test data: {args.test_data}")
    try:
        test_data = pd.read_parquet(args.test_data)
        print(f"✓ Loaded {len(test_data)} test samples\n")
    except Exception as e:
        print(f"❌ Unable to read test data: {e}")
        return
    
    # Run test
    try:
        result = test_model_on_dataset(
            args.model_name,
            test_data,
            output_dir,
            args.user_sim_urls,
            batch_size=args.batch_size,
            max_samples=args.n_samples,
            restart_config=restart_config
        )
        
        if result:
            print("\n✅ Testing completed!")
        else:
            print("\n❌ Testing failed")
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()