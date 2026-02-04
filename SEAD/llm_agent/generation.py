#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import json
import traceback
import subprocess
import time
import random
import threading

# ==================== Auto Path Configuration ====================
def _setup_import_paths():
    """Automatically configure import paths"""
    current_file = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file)
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    for path in [project_root, current_dir]:
        if path not in sys.path:
            sys.path.insert(0, path)

_setup_import_paths()

# ==================== Import Dependencies ====================
import torch
import re
import requests
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

try:
    from .tensor_helper import TensorHelper, TensorConfig
except (ImportError, ValueError):
    from tensor_helper import TensorHelper, TensorConfig

from verl import DataProto

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== User Response Cache ====================

class UserResponseCache:
    """
    Cache manager for User Simulator responses to accelerate training.
    
    Stage-specific hit rates:
    - Stage 0 (90%): Opening phase, highly homogeneous
    - Stage 1 (50%): Objection handling, diverse user behaviors
    - Stage 2 (20%): Cooperation phase, needs personalization
    - Stage 3 (90%): Confirmation step, ready to end dialogue
    """
    def __init__(self, 
                 cache_path: str, 
                 hit_rate: float = 0.5,
                 max_size_per_stage: int = 500, 
                 min_cache_size: int = 10,
                 max_total_size: int = 10000):
        self.cache_path = cache_path
        self.default_hit_rate = hit_rate
        self.max_size_per_stage = max_size_per_stage
        self.min_cache_size = min_cache_size
        self.max_total_size = max_total_size
        
        # Stage-specific hit rates (updated per requirements)
        self.stage_hit_rates = {
            '0': 0.9,  # 90% - Opening phase
            '1': 0.5,  # 50% - Objection handling
            '2': 0.2,  # 20% - Cooperation phase
            '3': 0.0,  # 0% - Confirmation step
        }
        
        # Track last cache usage per sample
        self.last_turn_used_cache: Dict[int, bool] = {}
        
        # Cache structure: {stage: [{'response': str, 'timestamp': float}, ...]}
        self.cache: Dict[str, List[Dict[str, Any]]] = {}
        self.lock = threading.Lock()
        self._load_cache()

    def _load_cache(self):
        """Load cache from disk (JSONL format)"""
        if os.path.exists(self.cache_path):
            try:
                count = 0
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            stage = str(data.get('stage', 'unknown'))
                            response = data.get('response')
                            timestamp = data.get('timestamp', time.time())
                            
                            if response and stage != 'unknown' and stage.isdigit():
                                if stage not in self.cache:
                                    self.cache[stage] = []
                                self.cache[stage].append({
                                    'response': response,
                                    'timestamp': timestamp
                                })
                                count += 1
                        except json.JSONDecodeError:
                            continue
                
                if count > self.max_total_size:
                    logger.warning(f"⚠️  Cache size ({count}) exceeds limit ({self.max_total_size}), cleaning up...")
                    self._cleanup_old_entries()
                
                total_count = sum(len(v) for v in self.cache.values())
                logger.info(f"✅ UserResponseCache loaded. Total entries: {total_count}. Stages: {list(self.cache.keys())}")
                logger.info(f"📊 Stage-specific hit rates: {self.stage_hit_rates}")
            except Exception as e:
                logger.error(f"Failed to load cache: {e}")

    def _cleanup_old_entries(self):
        """Remove oldest entries to maintain max_total_size limit"""
        total_count = sum(len(v) for v in self.cache.values())
        
        if total_count <= self.max_total_size:
            return
        
        all_entries = []
        for stage, entries in self.cache.items():
            for entry in entries:
                all_entries.append((stage, entry))
        
        all_entries.sort(key=lambda x: x[1]['timestamp'])
        num_to_remove = total_count - self.max_total_size
        
        removed_count = 0
        for stage, entry in all_entries[:num_to_remove]:
            if stage in self.cache and entry in self.cache[stage]:
                self.cache[stage].remove(entry)
                removed_count += 1
        
        empty_stages = [s for s, entries in self.cache.items() if not entries]
        for s in empty_stages:
            del self.cache[s]
        
        logger.info(f"🗑️  Cleaned up {removed_count} old entries. New total: {sum(len(v) for v in self.cache.values())}")
        self._save_cache()

    def _save_cache(self):
        """Save entire cache to disk (overwrites existing file)"""
        try:
            with open(self.cache_path, 'w', encoding='utf-8') as f:
                for stage, entries in self.cache.items():
                    for entry in entries:
                        f.write(json.dumps({
                            'stage': stage, 
                            'response': entry['response'],
                            'timestamp': entry['timestamp']
                        }, ensure_ascii=False) + '\n')
            logger.info(f"💾 Cache persisted to disk: {self.cache_path}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def get_response(self, stage: str, sample_id: int) -> Optional[str]:
        """
        Try to get a cached response.
        
        Rules:
        1. If last turn used cache for this sample, return None (force vLLM).
        2. If cache for this stage has fewer than min_cache_size entries, return None.
        3. Use stage-specific hit rate.
        4. If hit, mark this sample as "used cache" and return response.
        """
        stage = str(stage)
        
        with self.lock:
            # Rule 1: Prevent consecutive cache hits
            if self.last_turn_used_cache.get(sample_id, False):
                logger.debug(f"[Sample {sample_id}] Skipping cache (last turn used cache)")
                self.last_turn_used_cache[sample_id] = False
                return None
            
            # Rule 2: Check minimum cache size
            if stage not in self.cache or len(self.cache[stage]) < self.min_cache_size:
                return None
            
            # Rule 3: Stage-specific hit rate
            hit_rate = self.stage_hit_rates.get(stage, self.default_hit_rate)
            
            if random.random() < hit_rate:
                entry = random.choice(self.cache[stage])
                self.last_turn_used_cache[sample_id] = True
                logger.debug(f"[Sample {sample_id}] Cache HIT for stage {stage} (hit_rate={hit_rate})")
                return entry['response']
            else:
                logger.debug(f"[Sample {sample_id}] Cache MISS for stage {stage} (random > {hit_rate})")
                return None

    def add_response(self, stage: str, response: str):
        """Add a new response to cache with auto-cleanup"""
        stage = str(stage)
        
        if stage == 'unknown' or not stage.isdigit():
            logger.debug(f"⚠️  Skipped invalid stage: {stage}")
            return
        if not response or len(response) < 10:
            logger.debug(f"⚠️  Skipped short response (len={len(response)})")
            return

        with self.lock:
            if stage not in self.cache:
                self.cache[stage] = []
            
            existing = [e for e in self.cache[stage] if e['response'] == response]
            if existing:
                return
            
            if len(self.cache[stage]) >= self.max_size_per_stage:
                self.cache[stage].sort(key=lambda x: x['timestamp'])
                self.cache[stage].pop(0)
            
            new_entry = {
                'response': response,
                'timestamp': time.time()
            }
            self.cache[stage].append(new_entry)
            
            total_count = sum(len(v) for v in self.cache.values())
            if total_count > self.max_total_size:
                logger.info(f"📦 Cache full ({total_count}/{self.max_total_size}), triggering cleanup...")
                self._cleanup_old_entries()
            else:
                try:
                    with open(self.cache_path, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({
                            'stage': stage, 
                            'response': response,
                            'timestamp': new_entry['timestamp']
                        }, ensure_ascii=False) + '\n')
                except Exception as e:
                    logger.error(f"Failed to append to cache: {e}")

    def reset_sample_state(self, sample_id: int):
        """Reset cache usage flag for a sample (call at dialogue end)"""
        with self.lock:
            if sample_id in self.last_turn_used_cache:
                del self.last_turn_used_cache[sample_id]

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total = sum(len(v) for v in self.cache.values())
            per_stage = {stage: len(entries) for stage, entries in self.cache.items()}
            return {
                'total_entries': total,
                'num_stages': len(self.cache),
                'per_stage': per_stage,
                'max_total_size': self.max_total_size,
                'stage_hit_rates': self.stage_hit_rates
            }

# ==================== Configuration ====================

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int
    max_response_length: int
    max_obs_length: int
    no_think_rl: bool = False
    num_gpus: int = 1
    chatbot_url: str = "http://localhost:5000"
    request_timeout_s: int = 9999999
    mode: str = "call_chatbot"
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 40
    do_sample: bool = True
    max_api_retries: int = 3
    restart_wait_time: int = 9999999
    
    # === Cache Configuration ===
    use_cache: bool = True
    cache_hit_rate: float = 0.5
    cache_path: str = "./outputs/user_response_cache.jsonl"
    min_cache_size: int = 10
    max_total_size: int = 10000

# ==================== Manager ====================

class LLMGenerationManager:
    """Multi-turn Dialogue Rollout Manager"""

    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
        use_mock: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        self.use_mock = use_mock

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))
        
        self._init_special_tokens()
        
        # Initialize Cache
        self.response_cache = None
        if self.config.use_cache and not self.is_validation:
            os.makedirs(os.path.dirname(self.config.cache_path), exist_ok=True)
            self.response_cache = UserResponseCache(
                cache_path=self.config.cache_path,
                hit_rate=self.config.cache_hit_rate,
                min_cache_size=self.config.min_cache_size,
                max_total_size=self.config.max_total_size
            )
            logger.info(f"🚀 UserResponseCache initialized. Hit rate: {self.config.cache_hit_rate}, Min size: {self.config.min_cache_size}, Max total: {self.config.max_total_size}")

    def _init_special_tokens(self):
        """Pre-tokenize special tokens"""
        try:
            self.assistant_start_text = "<|im_end|>\n<|im_start|>assistant\n"
            self.assistant_start_ids = self.tokenizer(
                self.assistant_start_text,
                add_special_tokens=False,
                return_tensors='pt'
            )['input_ids']
            logger.info(f"✓ Pre-tokenized assistant_start: {self.assistant_start_ids.shape[1]} tokens")
        except Exception as e:
            logger.error(f"Failed to pre-tokenize special tokens: {e}")
            self.assistant_start_ids = None

    def _restart_vllm_server(self) -> bool:
        """Restart vLLM service"""
        print(f"\n{'='*40}\n🔄 [VeRL-Restart] Connection failure detected, preparing to restart vLLM...\n{'='*40}", flush=True)
        
        try:
            model_path = os.environ.get('VLLM_MODEL_PATH', 'your/path/to/Qwen2.5-14B-Instruct')
            
            try:
                port = self.config.chatbot_url.split(':')[-1].split('/')[0]
            except:
                port = os.environ.get('VLLM_PORT', '5000')
            
            mode = 'call_client' 
            run_id = f"auto_restart_{int(time.time())}"
            step = os.environ.get('VLLM_STEP', '0')
            script_path = os.path.abspath("./SEAD/vllm_service/start_vllm.sh")
            
            print(f"📋 Configuration confirmed: Port={port}, ID={run_id}", flush=True)
            print(f"🧹 [1/3] Performing safe cleanup...", flush=True)
            subprocess.run(f"fuser -k -9 {port}/tcp", shell=True, stderr=subprocess.DEVNULL)
            print("   -> Waiting 5 seconds for system to reclaim GPU memory...", flush=True)
            time.sleep(5) 

            cmd_str = f"bash {script_path} '{model_path}' '{run_id}' '{step}' '{mode}' '{port}'"
            full_cmd = ['bash', '-l', '-c', cmd_str]
            
            print(f"🚀 [2/3] Executing startup script...", flush=True)

            env = os.environ.copy()
            env['FLASK_ENV'] = 'production'
            if 'WERKZEUG_RUN_MAIN' in env: del env['WERKZEUG_RUN_MAIN']

            process = subprocess.Popen(full_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=os.getcwd(), env=env)

            print("📜 [Script Log Stream]:", flush=True)
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    print(f"   {line.strip()}", flush=True)

            rc = process.poll()

            if rc == 0:
                print(f"✅ [3/3] vLLM restart successful! Performing warmup test...", flush=True)
                try:
                    warmup_url = f"http://localhost:{port}/health"
                    warmup_data = {"prompt": "Hello", "max_tokens": 5, "n": 1, "temperature": 0.7}
                    resp = requests.post(warmup_url, json=warmup_data, timeout=60)
                    
                    if resp.status_code == 200:
                        print(f"✅ Warmup successful! Response: {resp.text[:100]}...", flush=True)
                        return True
                    else:
                        print(f"❌ Warmup failed (status code: {resp.status_code})", flush=True)
                        return False
                except Exception as e:
                    print(f"❌ Warmup request exception: {e}", flush=True)
                    return False
            else:
                print(f"❌ Restart failed (return code: {rc})", flush=True)
                return False

        except Exception as e:
            print(f"❌ Uncaught exception during restart: {e}", flush=True)
            traceback.print_exc()
            return False

    def _check_service_health(self) -> bool:
        """Check service health status"""
        try:
            resp = requests.get(f"{self.config.chatbot_url}/health", timeout=5)
            return resp.status_code == 200
        except:
            return False

    def _batch_tokenize(self, texts: List[str]) -> torch.Tensor:
        """Batch tokenization"""
        try:
            return self.tokenizer(texts, add_special_tokens=False, return_tensors='pt', padding="longest")['input_ids']
        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            raise

    def _cut_ids(self, ids: torch.Tensor, max_len: int) -> torch.Tensor:
        """Truncate token ids"""
        if ids.shape[1] > max_len:
            return ids[:, :max_len]
        return ids

    def _extract_response_blocks(self, outputs: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """Truncate to end tag"""
        try:
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            postprocessed = []
            
            if self.config.mode == 'call_chatbot':
                end_tag = '</status>'
            else:
                end_tag = '</user_status>'
            
            for s in decoded:
                if end_tag in s:
                    s = s.split(end_tag)[0] + end_tag
                postprocessed.append(s)
            return self._batch_tokenize(postprocessed), postprocessed
        except Exception as e:
            logger.error(f"Error extracting response blocks: {e}")
            raise

    def _parse_response_content(self, text: str) -> Tuple[bool, str]:
        """Extract content from LLM text"""
        if not isinstance(text, str):
            return False, ""
        try:
            if self.config.mode == 'call_client':
                m = re.search(r'<user_reply[^>]*>(.*?)</user_reply>', text, re.DOTALL | re.IGNORECASE)
                if m:
                    return True, m.group(1).strip()
                m = re.search(r'<reply[^>]*>(.*?)</reply>', text, re.DOTALL | re.IGNORECASE)
                if m:
                    return True, m.group(1).strip()
            else:
                m = re.search(r'<response[^>]*>(.*?)</response>', text, re.DOTALL | re.IGNORECASE)
                if m:
                    return True, m.group(1).strip()
        except Exception as e:
            logger.warning(f"Error parsing response content: {e}")
        return False, ""

    def _info_masked_concat(self, prompt: torch.Tensor, prompt_with_mask: torch.Tensor, response: torch.Tensor, info: torch.Tensor = None, pad_to_left: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Concatenate response and info while constructing info mask"""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]

        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device)
            tensors_with_mask.append(info_mask)

        concat = torch.cat(tensors, dim=1)
        concat_masked = torch.cat(tensors_with_mask, dim=1)

        mask = concat != pad_id if pad_to_left else concat == pad_id
        sorted_idx = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded = concat.gather(1, sorted_idx)
        padded_masked = concat_masked.gather(1, sorted_idx)
        return padded, padded_masked

    def _update_right_side(self, right: Dict[str, torch.Tensor], cur_responses: torch.Tensor, next_obs_ids: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Update accumulated responses on the right side"""
        if next_obs_ids is not None:
            responses, responses_masked = self._info_masked_concat(right['responses'], right['responses_with_info_mask'], cur_responses, next_obs_ids, pad_to_left=False)
        else:
            responses, responses_masked = self._info_masked_concat(right['responses'], right['responses_with_info_mask'], cur_responses, pad_to_left=False)
        eff_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, eff_len)
        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_masked[:, :max_len]}

    def _init_runtime_state(self, gen_batch: DataProto, batch_size: int):
        """Initialize runtime dialogue state for each sample"""
        meta = getattr(gen_batch, "meta_info", {}) or {}

        def _pull(key, default):
            v = meta.get(key, None)
            if v is None or (isinstance(v, list) and len(v) != batch_size):
                if callable(default):
                    return [default() for _ in range(batch_size)]
                else:
                    return [default for _ in range(batch_size)]
            return v

        histories = _pull('dialogue_history', list)
        turn_ids = _pull('turn_id', 0)
        batch_statuses = _pull('batch_statuses', list)

        for i in range(batch_size):
            if not isinstance(histories[i], list):
                histories[i] = []
            if not isinstance(batch_statuses[i], list):
                batch_statuses[i] = []

        return histories, turn_ids, batch_statuses

    def _batch_call_chatbot(self, server_url: str, histories: List[List[Dict[str, str]]], turn_ids: List[int], batch_statuses: List[List[int]]) -> List[Dict[str, Any]]:
        """Batch call user simulator API (with automatic restart)"""
        
        batch_size = len(histories)
        max_retries = self.config.max_api_retries
        
        logger.info(f"Calling batch API: {batch_size} samples (mode={self.config.mode})")
        
        vllm_mode = 'call_client' if self.config.mode == 'call_chatbot' else 'call_chatbot'
        
        batch_payload = {
            'batch_dialogue_histories': histories,
            'batch_turn_ids': turn_ids,
            'batch_statuses_list': batch_statuses,
            'mode': vllm_mode,
            'is_validation': self.is_validation
        }
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries}...")
                
                resp = requests.post(f'{server_url}/dialogue_batch', json=batch_payload, timeout=self.config.request_timeout_s)
                
                if resp.status_code == 200:
                    response_data = resp.json()
                    results = response_data.get('batch_results', [])
                    
                    while len(results) < batch_size:
                        idx = len(results)
                        results.append({'response': '', 'status_value': 0, 'turn_id': turn_ids[idx], 'batch_statuses': batch_statuses[idx], '_error': 'Missing response'})
                    
                    gen_time = response_data.get('generation_time_s', 0)
                    total_time = response_data.get('total_time_s', 0)
                    logger.info(f"✅ API call successful (attempt {attempt + 1}): Gen={gen_time:.2f}s, Total={total_time:.2f}s")
                    
                    return results
                
                else:
                    logger.warning(f"⚠️ HTTP {resp.status_code} (attempt {attempt + 1}/{max_retries})")
                    
                    if attempt < max_retries - 1:
                        logger.warning("Attempting to restart service...")
                        if self._restart_vllm_server():
                            logger.info(f"Waiting {self.config.restart_wait_time} seconds before retry...")
                            time.sleep(self.config.restart_wait_time)
                        else:
                            logger.error("Restart failed, waiting 5 seconds before retry...")
                            time.sleep(5)
            
            except requests.Timeout:
                logger.error(f"❌ Request timeout (attempt {attempt + 1}/{max_retries})")
                
                if attempt < max_retries - 1:
                    if not self._check_service_health():
                        logger.warning("Service health check failed, attempting restart...")
                        if self._restart_vllm_server():
                            logger.info(f"Waiting {self.config.restart_wait_time} seconds before retry...")
                            time.sleep(self.config.restart_wait_time)
                        else:
                            logger.error("Restart failed, waiting 5 seconds before retry...")
                            time.sleep(5)
                    else:
                        logger.info("Service still running, retrying directly...")
                        time.sleep(5)
            
            except requests.ConnectionError as e:
                logger.error(f"❌ Connection error: {e} (attempt {attempt + 1}/{max_retries})")
                
                if attempt < max_retries - 1:
                    logger.warning("Connection failed, attempting to restart service...")
                    if self._restart_vllm_server():
                        logger.info(f"Waiting {self.config.restart_wait_time} seconds before retry...")
                        time.sleep(self.config.restart_wait_time)
                    else:
                        logger.error("Restart failed, waiting 5 seconds before retry...")
                        time.sleep(5)
            
            except Exception as e:
                logger.error(f"❌ Unexpected error: {e} (attempt {attempt + 1}/{max_retries})")
                logger.error(traceback.format_exc())
                
                if attempt < max_retries - 1:
                    time.sleep(5)
        
        logger.error(f"❌ All {max_retries} attempts failed")
        return self._create_error_responses(batch_size, turn_ids, batch_statuses, error_msg=f"Service unavailable after {max_retries} retries")

    def _create_error_responses(self, count: int, turn_ids: List[int], batch_statuses: List[List[int]], error_msg: str) -> List[Dict[str, Any]]:
        """Create error response placeholders"""
        return [{'response': '', 'raw_response': '', 'turn_id': turn_ids[i], 'status': 'error', 'status_value': 0, 'batch_statuses': batch_statuses[i], 'batch_score': 0.0, 'completion_rate': 0.0, '_error': error_msg} for i in range(count)]

    def _example_level_pad(self, active_ids: torch.Tensor, active_texts: List[str], active_mask: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """Pad active sample results back to full batch"""
        batch_size = active_mask.shape[0]
        device = active_ids.device
        
        max_len = active_ids.shape[1]
        full_ids = torch.full((batch_size, max_len), self.tokenizer.pad_token_id, dtype=active_ids.dtype, device=device)
        full_ids[active_mask] = active_ids
        
        full_texts = [""] * batch_size
        active_idx = 0
        for i, is_active in enumerate(active_mask.tolist()):
            if is_active:
                full_texts[i] = active_texts[active_idx]
                active_idx += 1
        
        return full_ids, full_texts

    def _tokenize_observation(self, next_obs: List[str]) -> torch.Tensor:
        """Tokenize observation text"""
        processed_obs = []
        for obs in next_obs:
            if obs.strip():
                processed_obs.append(obs)
            else:
                processed_obs.append("<empty></empty>")
        
        try:
            ids = self.tokenizer(processed_obs, padding='longest', return_tensors='pt', add_special_tokens=False)['input_ids']
            return self._cut_ids(ids, self.config.max_obs_length)
        except Exception as e:
            logger.error(f"Error tokenizing observation: {e}")
            raise

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, next_obs_ids: torch.Tensor) -> DataProto:
        """Update rolling state"""
        try:
            new_input_ids = self.tensor_fn.concatenate_with_padding([rollings.batch['input_ids'], cur_responses, next_obs_ids])
            attn_mask = self.tensor_fn.create_attention_mask(new_input_ids)
            pos_ids = self.tensor_fn.create_position_ids(attn_mask)

            effective_len = attn_mask.sum(dim=1).max()
            max_len = min(self.config.max_prompt_length, effective_len)

            new_rollings = DataProto.from_dict({'input_ids': new_input_ids[:, -max_len:], 'position_ids': pos_ids[:, -max_len:], 'attention_mask': attn_mask[:, -max_len:]})
            new_rollings.meta_info.update(getattr(rollings, 'meta_info', {}) or {})
            return new_rollings
        except Exception as e:
            logger.error(f"Error updating rolling state: {e}")
            raise

    def _compose_final_output(self, left_side: Dict[str, torch.Tensor], right_side: Dict[str, torch.Tensor], meta_info: Dict[str, Any]) -> DataProto:
        """Compose final output"""
        try:
            final = right_side.copy()
            final['prompts'] = left_side['input_ids']
            final['input_ids'] = torch.cat([left_side['input_ids'], right_side['responses']], dim=1)

            left_attn = self.tensor_fn.create_attention_mask(left_side['input_ids'])
            right_attn = self.tensor_fn.create_attention_mask(right_side['responses'])
            right_masked_attn = self.tensor_fn.create_attention_mask(right_side['responses_with_info_mask'])

            final['attention_mask'] = torch.cat([left_attn, right_attn], dim=1)
            final['info_mask'] = torch.cat([left_attn, right_masked_attn], dim=1)
            final['position_ids'] = self.tensor_fn.create_position_ids(final['attention_mask'])

            out = DataProto.from_dict(final)
            out.meta_info.update(meta_info or {})
            return out
        except Exception as e:
            logger.error(f"Error composing final output: {e}")
            raise

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """If batch_size is not divisible by GPU count, pad by duplicating the first sample"""
        num_gpus = self.config.num_gpus or 1
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        bs = active_batch.batch['input_ids'].shape[0]
        rem = bs % num_gpus
        if rem == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        pad_n = num_gpus - rem
        
        try:
            padded = {}
            for k, v in active_batch.batch.items():
                pad_seq = v[0:1].repeat(pad_n, *([1] * (len(v.shape) - 1)))
                padded[k] = torch.cat([v, pad_seq], dim=0)
            padded_batch = DataProto.from_dict(padded)

            out = self.actor_rollout_wg.generate_sequences(padded_batch)

            trimmed = {k: v[:-pad_n] for k, v in out.batch.items()}
            out.batch = trimmed

            if hasattr(out, 'meta_info') and out.meta_info:
                meta_new = {}
                for k, v in out.meta_info.items():
                    if isinstance(v, torch.Tensor):
                        meta_new[k] = v[:-pad_n]
                    elif isinstance(v, list):
                        meta_new[k] = v[:-pad_n]
                    else:
                        meta_new[k] = v
                out.meta_info = meta_new
            
            return out
        except Exception as e:
            logger.error(f"Error in GPU padding: {e}")
            raise

    def run_llm_loop(self, gen_batch: DataProto, initial_input_ids: torch.Tensor) -> DataProto:
        """Multi-turn dialogue main loop"""
        batch_size = gen_batch.batch['input_ids'].shape[0]
        logger.info("="*70)
        logger.info(f"Starting LLM loop (Batch: {batch_size}, Max turns: {self.config.max_turns}, Mode: {self.config.mode})")
        logger.info("="*70)
        
        left = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        empty = initial_input_ids[:, []]
        right = {'responses': empty, 'responses_with_info_mask': empty}

        histories, turn_ids, statuses_history = self._init_runtime_state(gen_batch, batch_size)
        ground_truth_histories = [[] for _ in range(batch_size)]

        active_mask = torch.ones(batch_size, dtype=torch.bool)
        turns_stats = torch.zeros(batch_size, dtype=torch.int)
        active_counts = [active_mask.sum().item()]

        rollings = gen_batch

        # ====== Main Loop ======
        for step in range(self.config.max_turns):
            logger.info(f"\n--- Turn {step+1}/{self.config.max_turns} (Active: {active_mask.sum()}/{batch_size}) ---")
            
            if not active_mask.any():
                break

            rollings.batch = self.tensor_fn.cut_to_effective_len(rollings.batch, keys=['input_ids', 'attention_mask', 'position_ids'])
            rollings_active = DataProto.from_dict({k: v[active_mask] for k, v in rollings.batch.items()})

            try:
                gen_output = self._generate_with_gpu_padding(rollings_active)
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                break

            resp_ids, resp_texts_active = self._extract_response_blocks(gen_output.batch['responses'])
            resp_ids_full, resp_texts_full = self._example_level_pad(resp_ids, resp_texts_active, active_mask)

            valid_flags = []
            extracted_contents = []
            for txt in resp_texts_full:
                ok, content = self._parse_response_content(txt)
                valid_flags.append(ok)
                extracted_contents.append(content if ok else "")

            next_obs_list = [""] * batch_size
            dones = [0] * batch_size

            call_histories = []
            call_turn_ids = []
            call_batch_statuses = []
            call_indices = []
            
            step_results = {}

            for i, (is_active, is_valid) in enumerate(zip(active_mask.tolist(), valid_flags)):
                if not is_active: 
                    continue
                
                if is_valid:
                    actor_full_output = resp_texts_full[i]
                    
                    model_predicted_status = 0
                    if self.config.mode == 'call_chatbot':
                        status_match = re.search(r'<status[^>]*>(\d+)</status>', actor_full_output, re.IGNORECASE)
                        model_predicted_status = int(status_match.group(1)) if status_match else 0
                        
                        if model_predicted_status == 1:
                            dones[i] = 1
                            logger.info(f"[Sample {i}] Model predicted dialogue end (predicted_status=1)")

                    if self.config.mode == 'call_chatbot':
                        ground_truth_histories[i].append({'role': 'assistant', 'content': actor_full_output})
                    else:
                        ground_truth_histories[i].append({'role': 'user', 'content': actor_full_output})
                    
                    if self.config.mode == 'call_chatbot':
                        response_match = re.search(r'<response[^>]*>(.*?)</response>', actor_full_output, re.DOTALL | re.IGNORECASE)
                        
                        if response_match:
                            clean_response = response_match.group(1).strip()
                            clean_output = f"<response>{clean_response}</response>"
                        else:
                            clean_content = actor_full_output
                            for tag in ['cooperation_score', 'emotion_score', 'trust_score', 'noise_score', 'stage', 'think', 'status']:
                                clean_content = re.sub(f'<{tag}[^>]*>.*?</{tag}>', '', clean_content, flags=re.DOTALL | re.IGNORECASE)
                            clean_text = re.sub(r'<[^>]+>', '', clean_content).strip()
                            clean_output = f"<response>{clean_text}</response>" if clean_text else "<response></response>"
                    
                    else:
                        user_reply_match = re.search(r'<user_reply[^>]*>(.*?)</user_reply>', actor_full_output, re.DOTALL | re.IGNORECASE)
                        user_think_match = re.search(r'<user_think[^>]*>(.*?)</user_think>', actor_full_output, re.DOTALL | re.IGNORECASE)
                        
                        clean_output = ""
                        if user_think_match:
                            clean_output += f"<user_think>{user_think_match.group(1).strip()}</user_think>\n"
                        
                        if user_reply_match:
                            clean_output += f"<user_reply>{user_reply_match.group(1).strip()}</user_reply>"
                        else:
                            clean_content = actor_full_output
                            for tag in ['user_cooperation', 'user_emotion', 'user_trust', 'user_noise', 'user_status']:
                                clean_content = re.sub(f'<{tag}[^>]*>.*?</{tag}>', '', clean_content, flags=re.DOTALL | re.IGNORECASE)
                            if user_think_match:
                                clean_content = re.sub(r'<user_think[^>]*>.*?</user_think>', '', clean_content, flags=re.DOTALL | re.IGNORECASE)
                            clean_text = re.sub(r'<[^>]+>', '', clean_content).strip()
                            clean_output += f"<user_reply>{clean_text}</user_reply>" if clean_text else "<user_reply></user_reply>"
                    
                    if self.config.mode == 'call_chatbot':
                        histories[i].append({'role': 'assistant', 'content': clean_output})
                    else:
                        histories[i].append({'role': 'user', 'content': clean_output})

                    if model_predicted_status == 1:
                        continue
                    
                    # === 🚀 CACHE LOGIC START ===
                    current_stage = None
                    stage_match = re.search(r'<stage[^>]*>(\d+)</stage>', actor_full_output, re.IGNORECASE)
                    if stage_match:
                        current_stage = stage_match.group(1)
                    
                    cached_response = None
                    if self.response_cache and current_stage is not None:
                        cached_response = self.response_cache.get_response(current_stage, sample_id=i)  # ✅ 修复：传入 sample_id
                    
                    if cached_response:
                        cached_status = 0
                        status_match_cached = re.search(r'<user_status[^>]*>(-?\d+)</user_status>', cached_response)
                        if status_match_cached:
                            cached_status = int(status_match_cached.group(1))

                        step_results[i] = {'response': cached_response, 'raw_response': cached_response, 'status_value': cached_status, 'turn_id': turn_ids[i], 'batch_statuses': statuses_history[i], 'source': 'cache'}
                    else:
                        call_histories.append(histories[i].copy())
                        call_turn_ids.append(turn_ids[i])
                        call_batch_statuses.append(statuses_history[i].copy())
                        call_indices.append(i)
                        if current_stage is not None:
                            histories[i][-1]['_stage_for_cache'] = current_stage
                    # === 🚀 CACHE LOGIC END ===

                else:
                    logger.warning(f"[Sample {i}] Actor output format error, continuing dialogue")
                    
                    actor_full_output = resp_texts_full[i]
                    
                    if self.config.mode == 'call_chatbot':
                        ground_truth_histories[i].append({'role': 'assistant', 'content': actor_full_output})
                    else:
                        ground_truth_histories[i].append({'role': 'user', 'content': actor_full_output})
                    
                    error_msg = "<error>Format error, please output again.</error>"
                    
                    if self.config.mode == 'call_chatbot':
                        ground_truth_histories[i].append({'role': 'user', 'content': error_msg})
                    else:
                        ground_truth_histories[i].append({'role': 'assistant', 'content': error_msg})
                    
                    if self.config.mode == 'call_chatbot':
                        histories[i].append({'role': 'assistant', 'content': actor_full_output})
                        histories[i].append({'role': 'user', 'content': error_msg})
                    else:
                        histories[i].append({'role': 'user', 'content': actor_full_output})
                        histories[i].append({'role': 'assistant', 'content': error_msg})
                    
                    next_obs_list[i] = f"\n{error_msg}\n"

            if call_indices:
                logger.info(f"API Call needed for {len(call_indices)} samples (Cache Hit: {len(step_results)})")
                api_results = self._batch_call_chatbot(self.config.chatbot_url, call_histories, call_turn_ids, call_batch_statuses)
                
                for local_idx, idx_in_batch in enumerate(call_indices):
                    res = api_results[local_idx]
                    step_results[idx_in_batch] = res
                    
                    if self.response_cache and '_error' not in res:
                        stage_to_save = histories[idx_in_batch][-1].get('_stage_for_cache', None)
                        raw_resp = res.get('raw_response', res.get('response', ''))
                        
                        if stage_to_save is not None and raw_resp and len(raw_resp) > 10:
                            self.response_cache.add_response(stage_to_save, raw_resp)
            
            all_active_indices = list(step_results.keys())
            
            for idx_in_batch in all_active_indices:
                res = step_results[idx_in_batch]
                api_status = res.get('status', 'unknown')
                
                if '_error' in res or api_status == 'error':
                    logger.warning(f"[Sample {idx_in_batch}] API error: {res.get('_error', 'Unknown error')}, continuing dialogue")
                    
                    error_content = f"<error>API error: {res.get('_error', 'Unknown error')}</error>"
                    
                    if self.config.mode == 'call_chatbot':
                        ground_truth_histories[idx_in_batch].append({'role': 'user', 'content': error_content})
                    else:
                        ground_truth_histories[idx_in_batch].append({'role': 'assistant', 'content': error_content})
                    
                    if self.config.mode == 'call_chatbot':
                        histories[idx_in_batch].append({'role': 'user', 'content': error_content})
                    else:
                        histories[idx_in_batch].append({'role': 'assistant', 'content': error_content})
                    
                    next_obs_list[idx_in_batch] = "\nSystem error, please retry.\n"
                    continue
                
                status_value = int(res.get('status_value', 0))
                reply_text = res.get('response', '') or ''
                raw_response = res.get('raw_response', reply_text)
                
                real_cooperation = res.get('cooperation')
                real_emotion = res.get('emotion')
                real_trust = res.get('trust')
                real_noise = res.get('noise')
                
                if self.config.mode == 'call_chatbot':
                    user_reply_match = re.search(r'<user_reply[^>]*>(.*?)</user_reply>', raw_response, re.DOTALL | re.IGNORECASE)
                    user_think_match = re.search(r'<user_think[^>]*>(.*?)</user_think>', raw_response, re.DOTALL | re.IGNORECASE)
                    
                    user_reply_text = user_reply_match.group(1).strip() if user_reply_match else ""
                    user_think_text = user_think_match.group(1).strip() if user_think_match else ""
                    
                    if not user_reply_text:
                        logger.warning(f"[Sample {idx_in_batch}] user_reply is empty, continuing dialogue")
                        
                        gt_content = raw_response if raw_response.strip() else "<error>Empty user_reply</error>"
                        ground_truth_histories[idx_in_batch].append({'role': 'user', 'content': gt_content})
                        
                        histories[idx_in_batch].append({'role': 'user', 'content': gt_content})
                        
                        next_obs_list[idx_in_batch] = "\nUser did not reply, please continue.\n"
                        continue
                    
                    gt_content = ""
                    if user_think_text:
                        gt_content += f"<user_think>{user_think_text}</user_think>\n"
                    if user_reply_text:
                        gt_content += f"<user_reply>{user_reply_text}</user_reply>\n"
                    if real_cooperation is not None:
                        gt_content += f"<user_cooperation>{real_cooperation}</user_cooperation>\n"
                    if real_emotion is not None:
                        gt_content += f"<user_emotion>{real_emotion}</user_emotion>\n"
                    if real_trust is not None:
                        gt_content += f"<user_trust>{real_trust}</user_trust>\n"
                    if real_noise is not None:
                        gt_content += f"<user_noise>{real_noise}</user_noise>\n"
                    gt_content += f"<user_status>{status_value}</user_status>"
                    
                    ground_truth_histories[idx_in_batch].append({'role': 'user', 'content': gt_content})
                    histories[idx_in_batch].append({'role': 'user', 'content': gt_content})
                    
                    obs_content = ""
                    if user_think_text:
                        obs_content += f"<user_think>{user_think_text}</user_think>\n"
                    obs_content += f"<user_reply>{user_reply_text}</user_reply>"
                    next_obs_list[idx_in_batch] = f"\n{obs_content}\n"
                
                else:
                    response_matches = re.findall(r'<response[^>]*>(.*?)</response>', raw_response, re.DOTALL | re.IGNORECASE)
                    response_text = response_matches[-1].strip() if response_matches else ""
                    
                    if not response_text:
                        logger.warning(f"[Sample {idx_in_batch}] response is empty, continuing dialogue")
                        
                        gt_content = raw_response if raw_response.strip() else "<error>Empty response</error>"
                        ground_truth_histories[idx_in_batch].append({'role': 'assistant', 'content': gt_content})
                        
                        histories[idx_in_batch].append({'role': 'assistant', 'content': gt_content})
                        
                        next_obs_list[idx_in_batch] = "\nAgent did not reply, please continue.\n"
                        continue
                    
                    cooperation_score_match = re.search(r'<cooperation_score[^>]*>(\d+)</cooperation_score>', raw_response, re.IGNORECASE)
                    emotion_score_match = re.search(r'<emotion_score[^>]*>(\d+)</emotion_score>', raw_response, re.IGNORECASE)
                    trust_score_match = re.search(r'<trust_score[^>]*>(\d+)</trust_score>', raw_response, re.IGNORECASE)
                    noise_score_match = re.search(r'<noise_score[^>]*>(\d+)</noise_score>', raw_response, re.IGNORECASE)
                    stage_match_client = re.search(r'<stage[^>]*>(\d+)</stage>', raw_response, re.IGNORECASE)
                    think_match_client = re.search(r'<think[^>]*>(.*?)</think>', raw_response, re.DOTALL | re.IGNORECASE)
                    
                    cooperation_score = cooperation_score_match.group(1) if cooperation_score_match else None
                    emotion_score = emotion_score_match.group(1) if emotion_score_match else None
                    trust_score = trust_score_match.group(1) if trust_score_match else None
                    noise_score = noise_score_match.group(1) if noise_score_match else None
                    stage = stage_match_client.group(1) if stage_match_client else None
                    think_text = think_match_client.group(1).strip() if think_match_client else None
                    
                    gt_content = ""
                    if think_text:
                        gt_content += f"<think>{think_text}</think>\n"
                    gt_content += f"<response>{response_text}</response>\n"
                    if cooperation_score is not None:
                        gt_content += f"<cooperation_score>{cooperation_score}</cooperation_score>\n"
                    if emotion_score is not None:
                        gt_content += f"<emotion_score>{emotion_score}</emotion_score>\n"
                    if trust_score is not None:
                        gt_content += f"<trust_score>{trust_score}</trust_score>\n"
                    if noise_score is not None:
                        gt_content += f"<noise_score>{noise_score}</noise_score>\n"
                    if stage is not None:
                        gt_content += f"<stage>{stage}</stage>\n"
                    gt_content += f"<status>{status_value}</status>"
                    
                    ground_truth_histories[idx_in_batch].append({'role': 'assistant', 'content': gt_content})
                    
                    obs_content = ""
                    if think_text:
                        obs_content += f"<think>{think_text}</think>\n"
                    obs_content += f"<response>{response_text}</response>\n"
                    if cooperation_score is not None:
                        obs_content += f"<cooperation_score>{cooperation_score}</cooperation_score>\n"
                    if emotion_score is not None:
                        obs_content += f"<emotion_score>{emotion_score}</emotion_score>\n"
                    if trust_score is not None:
                        obs_content += f"<trust_score>{trust_score}</trust_score>\n"
                    if noise_score is not None:
                        obs_content += f"<noise_score>{noise_score}</noise_score>\n"
                    if stage is not None:
                        obs_content += f"<stage>{stage}</stage>"
                    
                    histories[idx_in_batch].append({'role': 'assistant', 'content': obs_content.strip()})
                    
                    next_obs_content = f"<response>{response_text}</response>"
                    next_obs_list[idx_in_batch] = f"\n{next_obs_content}\n"
                
                turn_ids[idx_in_batch] = int(res.get('turn_id', turn_ids[idx_in_batch]))
                statuses_history[idx_in_batch] = res.get('batch_statuses', statuses_history[idx_in_batch]) or []
                
                if status_value == 1 or status_value == -1:
                    dones[idx_in_batch] = 1
                    logger.info(f"[Sample {idx_in_batch}] Dialogue terminated, status_value={status_value}")

            turns_stats[active_mask] += 1
            next_obs_ids = self._tokenize_observation(next_obs_list)

            rollings = self._update_rolling_state(rollings, resp_ids_full, next_obs_ids)
            right = self._update_right_side(right, resp_ids_full, next_obs_ids)

            new_active = torch.tensor([not bool(d) for d in dones], dtype=torch.bool)
            active_mask = active_mask & new_active
            active_counts.append(active_mask.sum().item())

        # ====== Cache Statistics ======
        if self.response_cache:
            stats = self.response_cache.get_stats()
            logger.info(f"📦 Cache Stats: {stats['total_entries']}/{stats['max_total_size']} entries across {stats['num_stages']} stages")
            if stats['total_entries'] > stats['max_total_size'] * 0.9:
                logger.warning(f"⚠️  Cache is {stats['total_entries']/stats['max_total_size']*100:.1f}% full")

        # ====== Reset Cache State ======
        if self.response_cache:
            for i in range(batch_size):
                self.response_cache.reset_sample_state(i)

        # ====== Final Round Generation ======
        if active_mask.any():
            logger.info(f"\n[Timeout Handling] {active_mask.sum().item()} dialogues reached max turns but not completed")
            
            rollings.batch = self.tensor_fn.cut_to_effective_len(rollings.batch, keys=['input_ids', 'attention_mask', 'position_ids'])
            rollings_active = DataProto.from_dict({k: v[active_mask] for k, v in rollings.batch.items()})

            try:
                gen_output = self._generate_with_gpu_padding(rollings_active)
                resp_ids, resp_texts_active = self._extract_response_blocks(gen_output.batch['responses'])
                resp_ids_full, resp_texts_full = self._example_level_pad(resp_ids, resp_texts_active, active_mask)
                
                for i, (is_active, text) in enumerate(zip(active_mask.tolist(), resp_texts_full)):
                    if is_active and text.strip():
                        if self.config.mode == 'call_chatbot':
                            ground_truth_histories[i].append({'role': 'assistant', 'content': text})
                        else:
                            ground_truth_histories[i].append({'role': 'user', 'content': text})
                
                right = self._update_right_side(right, cur_responses=resp_ids_full, next_obs_ids=None)
            except Exception as e:
                logger.error(f"Final generation failed: {e}")

        # ====== Compose Final Output ======
        meta_info = getattr(gen_output, 'meta_info', {}) or {}
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_traj_counts'] = active_counts
        meta_info['dialogue_history'] = histories
        meta_info['ground_truth_history'] = ground_truth_histories
        meta_info['turn_id'] = turn_ids
        meta_info['batch_statuses'] = statuses_history

        logger.info("\n" + "="*70)
        logger.info("LLM loop completed")
        logger.info(f"  Turns stats: {turns_stats.tolist()}")
        logger.info("="*70 + "\n")

        # Save rollout data
        try:
            save_dir = "./outputs/temp_dialog"
            os.makedirs(save_dir, exist_ok=True)
            jsonl_path = os.path.join(save_dir, "rollout_data.jsonl")
            
            with open(jsonl_path, 'a', encoding='utf-8') as f:
                for i in range(batch_size):
                    sample_data = {'turns': int(turns_stats[i]), 'dialogue_history': histories[i], 'ground_truth_history': ground_truth_histories[i], 'batch_statuses': statuses_history[i], 'turn_id': turn_ids[i]}
                    f.write(json.dumps(sample_data, ensure_ascii=False) + '\n')
            
            logger.info(f"✅ Saved {batch_size} rollout samples to {jsonl_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save rollout data: {e}")
            logger.error(traceback.format_exc())

        final = self._compose_final_output(left, right, meta_info)
        return final