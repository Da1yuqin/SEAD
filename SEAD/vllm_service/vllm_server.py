#!/usr/bin/env python
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
import vllm
import argparse
import json
import os
import time
import torch
from transformers import AutoTokenizer
import re
import traceback
import logging
from typing import List, Dict, Any
from pathlib import Path
import sys
import random

# ----------------------- Logging Configuration ----------------------- #
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ------------------------- Command-Line Arguments ------------------------- #
parser = argparse.ArgumentParser(description='Multi-turn Dialogue Evaluation Service')
parser.add_argument('--port', type=str, default='5000', help='Service port')
parser.add_argument('--model_path', type=str, required=True, help='Path to the model')
parser.add_argument('--gpu_mem_util', type=float, default=0.90, help='GPU memory utilization fraction')
parser.add_argument('--mode', type=str, choices=['call_client', 'call_chatbot'], required=True, help='Mode')
parser.add_argument('--prompt_dir', type=str, default='./verl/trainer/config/format_prompt', help='Prompt dir')
parser.add_argument('--max_model_len', type=int, default=30000, help='Context length')
parser.add_argument('--max_num_seqs', type=int, default=256, help='Max sequences')
parser.add_argument('--dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16', 'float32'])
parser.add_argument('--max_gen_tokens', type=int, default=512, help='Max gen tokens')
parser.add_argument('--tensor_parallel_size', type=int, default=2, help='TP size')
parser.add_argument('--user_params_dir', type=str, default='./outputs/temp', help='User params directory')
parser.add_argument('--max_num_batched_tokens', type=int, default=None, help='Max batched tokens')
parser.add_argument('--disable-custom-all-reduce', action='store_true', help='Disable custom all reduce')
parser.add_argument('--enforce-eager', action='store_true', help='Enforce eager mode')

args = parser.parse_args()


# ----------------------- User Params Management ----------------------- #

class UserParamsManager:
    """Manage user_params dataset and index"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load datasets
        self.train_user_params = self._load_user_params("train")
        self.test_user_params = self._load_user_params("test")
        
        # Load indexes
        self.train_index = self._load_index("train")
        self.test_index = self._load_index("test")
        
        logger.info(f"✅ Loaded {len(self.train_user_params)} train user_params")
        logger.info(f"✅ Loaded {len(self.test_user_params)} test user_params")
        logger.info(f"✅ Train index: {self.train_index}, Test index: {self.test_index}")
    
    def _load_user_params(self, split: str) -> list:
        """Load user_params"""
        file_path = self.data_dir / f"{split}_user_params.jsonl"
        if not file_path.exists():
            logger.warning(f"⚠️  {file_path} not found")
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
    
    def get_batch_user_params(self, batch_size: int, is_validation: bool = False) -> list:
        """Get user_params for current batch"""
        if is_validation:
            params_list = self.test_user_params
            current_index = self.test_index
            split = "test"
        else:
            params_list = self.train_user_params
            current_index = self.train_index
            split = "train"
        
        if len(params_list) == 0:
            logger.warning(f"⚠️  {split} user_params is empty")
            return [{}] * batch_size
        
        # Auto reset: If index exceeds range, reset to 0
        if current_index >= len(params_list):
            logger.info(f"✅ {split} dataset exhausted (index={current_index}), resetting to 0")
            current_index = 0
            if is_validation:
                self.test_index = 0
            else:
                self.train_index = 0
            self._save_index(split, 0)
        
        # Calculate actual available samples
        end_index = min(current_index + batch_size, len(params_list))
        actual_batch_size = end_index - current_index
        
        batch_params = params_list[current_index:end_index]
        
        # If less than batch_size, pad with last sample
        while len(batch_params) < batch_size:
            if batch_params:
                batch_params.append(batch_params[-1])
            else:
                batch_params.append({})
        
        # Update index
        new_index = end_index
        if is_validation:
            self.test_index = new_index
        else:
            self.train_index = new_index
        self._save_index(split, new_index)
        
        logger.debug(f"[{split}] Sampled {actual_batch_size} params (index: {current_index}->{new_index}/{len(params_list)})")
        
        return batch_params

# Initialize UserParamsManager
user_params_manager = UserParamsManager(args.user_params_dir)

# ----------------------- Prompt Loading ----------------------- #

def load_prompt_file(prompt_dir: str, mode: str) -> str:
    try:
        filename = f"{mode}.txt"
        prompt_file = os.path.join(prompt_dir, filename)
        if not os.path.exists(prompt_file):
            if mode == 'call_client': prompt_file = os.path.join(prompt_dir, 'call_client.txt')
            elif mode == 'call_chatbot': prompt_file = os.path.join(prompt_dir, 'call_chatbot.txt')
        
        if not os.path.exists(prompt_file):
            return "You are a helpful assistant."
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # Don't remove placeholders, keep them for subsequent filling
            return content
    except Exception as e:
        logger.error(f"Failed to load prompt file: {e}")
        return "You are a helpful assistant."

BASE_SYSTEM_PROMPT_TEMPLATE = load_prompt_file(args.prompt_dir, args.mode)

# ------------------------- vLLM Initialization ------------------------ #
logger.info('='*70)
logger.info('Initializing vLLM Service (Chat Template Mode)')
logger.info(f'Model: {args.model_path}')
logger.info('='*70)

try:
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
        logger.warning('⚠️  Tokenizer does not have chat_template, will use fallback mode')
        USE_CHAT_TEMPLATE = False
    else:
        logger.info(f'✓ Detected chat_template: {tokenizer.chat_template[:100]}...')
        USE_CHAT_TEMPLATE = True
    
    model = vllm.LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_mem_util,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens or args.max_model_len,  # Use parameter
        enforce_eager=args.enforce_eager,  # Use parameter
        disable_custom_all_reduce=args.disable_custom_all_reduce,  # Use parameter
        distributed_executor_backend='ray',
    )
except Exception as e:
    logger.critical(f'Failed to load vLLM model: {e}')
    exit(1)

stop_words = [] 

sample_params = vllm.SamplingParams(
    max_tokens=args.max_gen_tokens,
    temperature=0.9,
    top_p=0.95,
    top_k=40,
    stop=stop_words,
    stop_token_ids=[tokenizer.eos_token_id] if hasattr(tokenizer, 'eos_token_id') else None,
    n=1,
)

def fill_user_params(template: str, user_params: Dict[str, Any]) -> str:
    """Fill user_params placeholders (supports ${...} and {...} formats)"""
    try:
        filled = template
        replaced_count = 0
        
        # Field mapping: user_params field -> template placeholder
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
            placeholders_to_try.append(f"${{{key}}}")   # ${cooperation}
            placeholders_to_try.append(f"{{{key}}}")    # {cooperation}
            
            # Mapped field name
            mapped_key = field_mapping.get(key)
            if mapped_key:
                placeholders_to_try.append(f"${{{mapped_key}}}")  # ${initial_cooperation}
                placeholders_to_try.append(f"{{{mapped_key}}}")   # {initial_cooperation}
            
            # Try replacement
            for placeholder in placeholders_to_try:
                if placeholder in filled:
                    filled = filled.replace(placeholder, value_str)
                    replaced_count += 1
                    if random.random() < 0.05:
                        logger.info(f"[DEBUG] ✅ Replaced {placeholder}")
                    break
        
        # Debug logs
        if random.random() < 0.05:
            unfilled_dollar = re.findall(r'\$\{([^}]+)\}', filled)
            unfilled_brace = re.findall(r'\{([^}]+)\}', filled)
            
            if unfilled_dollar:
                logger.warning(f"[DEBUG] ⚠️  Unfilled ${{{...}}}: {unfilled_dollar}")
            if unfilled_brace:
                logger.warning(f"[DEBUG] ⚠️  Unfilled {{...}}: {unfilled_brace}")
            
            logger.info(f"[DEBUG] ✅ Replaced {replaced_count}/{len(user_params)} placeholders")
        
        # Remove unfilled placeholders
        filled = re.sub(r'\$\{[^}]+\}', '', filled)  # Remove ${...}
        filled = re.sub(r'\{[^}]+\}', '', filled)    # Remove {...}
        
        return filled
    except Exception as e:
        logger.error(f"❌ Error filling user_params: {e}")
        logger.error(traceback.format_exc())
        return template

def build_chat_messages(
    base_system_prompt_template: str, 
    dialogue_history: List[Dict[str, str]], 
    user_params: Dict[str, Any],  
    mode: str = 'call_chatbot'
) -> List[Dict[str, str]]:
    '''Build chat messages'''
    
    # 1. Fill user_params
    base_system_prompt = fill_user_params(base_system_prompt_template, user_params)
    
    # 2. Build dialogue history
    history_text = ""
    
    if random.random() < 0.01:
        print("dialogue_history", dialogue_history)

    if dialogue_history:
        for item in dialogue_history:
            role = item.get('role', 'unknown')
            content = item.get('content', '')
            
            if not content: 
                continue
            
            if mode == 'call_client':
                if role == 'assistant':
                    r_match = re.search(r'<response[^>]*>(.*?)</response>', content, re.DOTALL | re.IGNORECASE)
                    if r_match:
                        resp_text = r_match.group(1).strip()
                        history_text += f"\n<response>{resp_text}</response>"
                    else:
                        clean_text = re.sub(r'<[^>]+>', '', content).strip()
                        if clean_text:
                            history_text += f"\n<response>{clean_text}</response>"

                elif role == 'user':
                    history_text += f"\n{content.strip()}"

            elif mode == 'call_chatbot':
                if role == 'user':
                    user_reply_match = re.search(r'<user_reply[^>]*>(.*?)</user_reply>', content, re.DOTALL | re.IGNORECASE)
                    
                    if user_reply_match:
                        user_reply_text = user_reply_match.group(1).strip()
                        history_text += f"\n<user_reply>{user_reply_text}</user_reply>"
                    else:
                        clean_content = content
                        for tag in ['user_think', 'user_cooperation', 'user_emotion', 'user_trust', 'user_noise', 'user_status']:
                            clean_content = re.sub(rf'<{tag}[^>]*>.*?</{tag}>', '', clean_content, flags=re.DOTALL | re.IGNORECASE)
                        
                        clean_text = re.sub(r'<[^>]+>', '', clean_content).strip()
                        if clean_text:
                            history_text += f"\n<user_reply>{clean_text}</user_reply>"
                
                elif role == 'assistant':
                    history_text += f"\n{content.strip()}"

    # 3. Concatenate Prompt
    if mode == 'call_client':
        instruction = "Below is the dialogue history, where content inside `<response></response>` is the reply from the other party (customer service). Please output your reply according to the output format:"
    else:
        instruction = "Below is the dialogue history, where content inside `<user_reply></user_reply>` is the reply from the other party (user). Please output your reply according to the output format:"
    
    full_system_content = f"{base_system_prompt}\n\n## Start Dialogue\n{instruction}\n{history_text}\n\nPlease output your reply for this turn (follow the output format above)."
    
    messages = [
        {"role": "system", "content": full_system_content}
    ]
    
    return messages


def build_prompt_with_chat_template(messages: List[Dict[str, str]]) -> str:
    try:
        if USE_CHAT_TEMPLATE:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt
        else:
            return build_prompt_fallback(messages)
    except Exception as e:
        logger.warning(f'apply_chat_template failed: {e}, using fallback mode')
        return build_prompt_fallback(messages)

def build_prompt_fallback(messages: List[Dict[str, str]]) -> str:
    prompt = ""
    for msg in messages:
        role = msg['role']
        content = msg['content']
        if role == 'system':
            prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == 'user':
            prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == 'assistant':
            prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt

# ---------------------- Response Parsing ----------------------- #

def extract_first_response_block(text: str, mode: str = 'call_chatbot') -> str:
    try:
        if mode == 'call_chatbot':
            pattern = r'(<response[^>]*>.*?</response>.*?<status[^>]*>.*?</status>)'
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                return matches[-1].strip()

            r_matches = re.findall(r'<response[^>]*>(.*?)</response>', text, re.DOTALL | re.IGNORECASE)
            if r_matches:
                last_resp = r_matches[-1]
                if "<status>" in text: return text
                return f"<response>{last_resp}</response>\n<status>0</status>"

        elif mode == 'call_client':
            pattern = r'(<user_reply[^>]*>.*?</user_reply>.*?<user_status[^>]*>.*?</user_status>)'
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                return matches[-1].strip()

            r_matches = re.findall(r'<user_reply[^>]*>(.*?)</user_reply>', text, re.DOTALL | re.IGNORECASE)
            if r_matches:
                last_resp = r_matches[-1]
                if "<user_status>" in text: return text
                return f"<user_reply>{last_resp}</user_reply>\n<user_status>0</user_status>"

        return text
    except Exception:
        return text

def parse_model_output(text: str, mode: str = 'call_chatbot') -> Dict[str, Any]:
    response = ""
    status = 0
    cooperation = None
    emotion = None
    trust = None
    noise = None
    stage = None
    think = ""

    try:
        if mode == 'call_chatbot':
            r_matches = re.findall(r'<response[^>]*>(.*?)</response>', text, re.DOTALL | re.IGNORECASE)
            if r_matches: 
                response = r_matches[-1].strip()
            
            th_m = re.search(r'<think[^>]*>(.*?)</think>', text, re.DOTALL | re.IGNORECASE)
            if th_m: 
                think = th_m.group(1).strip()

            s_matches = re.findall(r'<status[^>]*>(-?\d+)</status>', text, re.IGNORECASE)
            if s_matches: 
                status = int(s_matches[-1])

            c_m = re.search(r'<cooperation_score[^>]*>(\d+)</cooperation_score>', text, re.IGNORECASE)
            if c_m: 
                cooperation = int(c_m.group(1))
            
            e_m = re.search(r'<emotion_score[^>]*>(\d+)</emotion_score>', text, re.IGNORECASE)
            if e_m: 
                emotion = int(e_m.group(1))
            
            t_m = re.search(r'<trust_score[^>]*>(\d+)</trust_score>', text, re.IGNORECASE)
            if t_m: 
                trust = int(t_m.group(1))
            
            n_m = re.search(r'<noise_score[^>]*>(\d+)</noise_score>', text, re.IGNORECASE)
            if n_m: 
                noise = int(n_m.group(1))
            
            st_m = re.search(r'<stage[^>]*>(\d+)</stage>', text, re.IGNORECASE)
            if st_m: 
                stage = int(st_m.group(1))

        elif mode == 'call_client':
            r_matches = re.findall(r'<user_reply[^>]*>(.*?)</user_reply>', text, re.DOTALL | re.IGNORECASE)
            if r_matches: 
                response = r_matches[-1].strip()

            th_m = re.search(r'<user_think[^>]*>(.*?)</user_think>', text, re.DOTALL | re.IGNORECASE)
            if th_m: 
                think = th_m.group(1).strip()

            s_matches = re.findall(r'<user_status[^>]*>(-?\d+)</user_status>', text, re.IGNORECASE)
            if s_matches: 
                status = int(s_matches[-1])

            c_m = re.search(r'<user_cooperation[^>]*>(\d+)</user_cooperation>', text, re.IGNORECASE)
            if c_m: 
                cooperation = int(c_m.group(1))
            
            e_m = re.search(r'<user_emotion[^>]*>(\d+)</user_emotion>', text, re.IGNORECASE)
            if e_m: 
                emotion = int(e_m.group(1))
            
            t_m = re.search(r'<user_trust[^>]*>(\d+)</user_trust>', text, re.IGNORECASE)
            if t_m: 
                trust = int(t_m.group(1))
            
            n_m = re.search(r'<user_noise[^>]*>(\d+)</user_noise>', text, re.IGNORECASE)
            if n_m: 
                noise = int(n_m.group(1))

    except Exception as e:
        logger.error(f"Error parsing output in mode {mode}: {e}")

    return {
        'response': response,
        'status': status,
        'cooperation': cooperation,
        'emotion': emotion,
        'trust': trust,
        'noise': noise,
        'stage': stage,
        'think': think,
        'raw_text': text
    }

# ---------------------------- Flask Application --------------------------- #
app = Flask(__name__)

@app.route('/dialogue_batch', methods=['POST'])
def dialogue_batch():
    request_start_time = time.time()
    
    try:
        request_data = request.get_json()
        if not request_data: return jsonify({'error': 'Empty body'}), 400
        
        request_mode = request_data.get('mode', args.mode)
        if request_mode not in ('call_chatbot', 'call_client'):
            request_mode = args.mode

        is_validation = request_data.get('is_validation', False)  

        histories = request_data.get('batch_dialogue_histories', [])
        turn_ids = request_data.get('batch_turn_ids', [])
        statuses_list = request_data.get('batch_statuses_list', [])
        
        batch_size = len(histories)
        if batch_size == 0: return jsonify({'error': 'Empty batch'}), 400
        
        logger.info(f'[Batch] Processing {batch_size} samples (request_mode={request_mode}, is_validation={is_validation})')
        
        # 1. Get user_params
        batch_user_params = user_params_manager.get_batch_user_params(batch_size, is_validation)
        
        # 2. Build Prompts (fill user_params)
        all_prompts = []
        for h, user_params in zip(histories, batch_user_params):
            messages = build_chat_messages(BASE_SYSTEM_PROMPT_TEMPLATE, h, user_params, mode=request_mode)
            prompt = build_prompt_with_chat_template(messages)
            if random.random() < 0.01:
                print("【VLLM】prompt", prompt[800:])
            all_prompts.append(prompt)

        # 3. vLLM generation
        gen_start = time.time()
        try:
            responses = model.generate(all_prompts, sampling_params=sample_params, use_tqdm=False)
        except Exception as e:
            logger.error(f"vLLM error: {e}")
            return jsonify({'error': str(e)}), 500
        gen_time = time.time() - gen_start

        # 4. Parse results
        batch_results = []
        for idx, resp in enumerate(responses):
            turn_id = turn_ids[idx] if idx < len(turn_ids) else 0
            current_statuses = statuses_list[idx] if idx < len(statuses_list) else []
            
            try:
                raw_text = resp.outputs[0].text.strip()
                
                assistant_marker = "<|im_start|>assistant\n"
                if assistant_marker in raw_text:
                    raw_text = raw_text.split(assistant_marker)[-1].strip()
                
                raw_text = raw_text.replace("<|im_end|>", "").strip()

                # Smart completion of closing tags
                if request_mode == 'call_chatbot':
                    if "<response>" in raw_text and "</response>" not in raw_text:
                        raw_text += "</response>"
                    
                    if "<status>" in raw_text:
                        if not re.search(r'</status>', raw_text):
                             raw_text += "</status>"
                    else:
                        if "<response>" in raw_text:
                            raw_text += "\n<status>0</status>"
                
                elif request_mode == 'call_client':
                    if "<user_reply>" in raw_text and "</user_reply>" not in raw_text:
                        raw_text += "</user_reply>"
                    
                    if "<user_status>" in raw_text:
                        if not re.search(r'</user_status>', raw_text):
                            raw_text += "</user_status>"
                    else:
                        if "<user_reply>" in raw_text:
                            raw_text += "\n<user_status>0</user_status>"
                
                clean_text = extract_first_response_block(raw_text, mode=request_mode)
                parsed = parse_model_output(raw_text, mode=request_mode)
                
                status_val = parsed['status']
                new_statuses = current_statuses + [status_val]
                
                valid = [s for s in new_statuses if s != -1]
                score = sum(valid)/len(valid) if valid else 0.0
                comp_rate = sum(1 for s in valid if s==1)/len(valid) if valid else 0.0
                
                batch_results.append({
                    'response': parsed['response'],
                    'raw_response': clean_text,
                    'turn_id': turn_id + 1,
                    'status': 'success',
                    'status_value': status_val,
                    'batch_statuses': new_statuses,
                    'batch_score': score,
                    'completion_rate': comp_rate,
                    'cooperation': parsed.get('cooperation'),
                    'emotion': parsed.get('emotion'),
                    'trust': parsed.get('trust'),
                    'noise': parsed.get('noise'),
                })
            except Exception as e:
                logger.error(f"Parse error idx {idx}: {e}")
                batch_results.append({
                    'response': '', 'status_value': 0, 'turn_id': turn_id,
                    'batch_statuses': current_statuses, '_error': str(e)
                })

        total_time = time.time() - request_start_time
        logger.info(f'[Batch] Done. Gen: {gen_time:.2f}s, Total: {total_time:.2f}s')
        
        return jsonify({
            'batch_results': batch_results,
            'generation_time_s': round(gen_time, 2),
            'total_time_s': round(total_time, 2)
        })

    except Exception as e:
        logger.error(f"Server error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'gpu': torch.cuda.is_available()}), 200

if __name__ == '__main__':
    logger.info(f'Starting vLLM Server on port {args.port}...')
    app.run(host='0.0.0.0', port=int(args.port), threaded=True, debug=False)