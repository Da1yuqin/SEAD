#!/usr/bin/env python3
# generate_profiles_parallel.py
# -*- coding: utf-8 -*-
"""
Parallel User Profile Generation with Anti-Hallucination Validation
"""

import argparse
import json
import os
import re
import random
import time
import fcntl
import subprocess
import traceback
from typing import Dict, List, Tuple, Set
from pathlib import Path
from multiprocessing import Process, Queue, Manager
from collections import Counter

# ==================== Resource Cleanup ==================== #

def force_cleanup():
    try:
        subprocess.run("pkill -9 -f 'vllm'", shell=True, stderr=subprocess.DEVNULL)
        subprocess.run("pkill -9 -f 'ray'", shell=True, stderr=subprocess.DEVNULL)
        subprocess.run("rm -rf /tmp/ray_*", shell=True, stderr=subprocess.DEVNULL)
        subprocess.run("rm -rf /dev/shm/ray_*", shell=True, stderr=subprocess.DEVNULL)
    except: pass

# ==================== Data Loading ==================== #

def load_analysis(analysis_file: str) -> Dict:
    if not os.path.exists(analysis_file):
        return {}
    with open(analysis_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_behavior_library(library_file: str) -> Dict:
    """Load behavior library"""
    p = Path(library_file)
    if not p.exists():
        return {"all_actions": [], "categories": {}, "category_actions": {}}
    
    with open(p, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Handle JSONL format if necessary
        lines = content.strip().split('\n')
        if lines:
            try:
                data = json.loads(lines[0])
            except:
                # Fallback structure
                return {"all_actions": [], "categories": {}, "category_actions": {}}
        else:
            return {"all_actions": [], "categories": {}, "category_actions": {}}
    
    categories = data.get("User_Persona_Classification", {}) # Try English key first
    if not categories:
        categories = data.get("用户画像分类", {}) # Try Chinese key
        
    all_actions = []
    category_actions = {}
    
    for category, actions in categories.items():
        if isinstance(actions, list) and actions:
            all_actions.extend(actions)
            category_actions[category] = actions
    
    # Deduplicate all actions
    all_actions = list(set(all_actions))
    
    return {
        "all_actions": all_actions,
        "categories": categories,
        "category_actions": category_actions
    }

def sample_random_profile(
    behavior_lib: Dict,
    rng: random.Random,
    analysis: Dict = None,
    min_behaviors: int = 1,
    max_behaviors: int = 3
) -> Tuple[int, int, int, List[str]]:
    """Generate profile based on weights"""
    
    if analysis is None or 'recommended_weights' not in analysis:
        # Default weights (avoid extremes)
        cooperation_weights = [0, 0.33, 0.34, 0.33, 0]
        emotion_weights = [0, 0.5, 0.5, 0]
        trust_weights = [0, 0.25, 0.25, 0.25, 0.25, 0]
    else:
        recommended = analysis['recommended_weights']
        cooperation_weights = recommended['cooperation_weights']
        emotion_weights = recommended['emotion_weights']
        trust_weights = recommended['trust_weights']
    
    # Sampling
    cooperation = rng.choices(range(5), weights=cooperation_weights)[0]
    emotion = rng.choices(range(4), weights=emotion_weights)[0]
    trust = rng.choices(range(6), weights=trust_weights)[0]
    
    # Retry logic for extremes
    max_retries = 10
    retry_count = 0
    while (cooperation in [0, 4] or emotion in [0, 3] or trust in [0, 5]) and retry_count < max_retries:
        cooperation = rng.choices(range(5), weights=cooperation_weights)[0]
        emotion = rng.choices(range(4), weights=emotion_weights)[0]
        trust = rng.choices(range(6), weights=trust_weights)[0]
        retry_count += 1
    
    if retry_count >= max_retries:
        cooperation, emotion, trust = 2, 1, 2
    
    # Behavior extraction
    all_actions = behavior_lib.get('all_actions', [])
    if not all_actions:
        behaviors = ['No specific behavior']
    else:
        num_behaviors = rng.randint(min_behaviors, min(max_behaviors, len(all_actions)))
        behaviors = rng.sample(all_actions, num_behaviors)
    
    return cooperation, emotion, trust, behaviors

# ==================== Prompt Construction (Anti-Hallucination) ==================== #

def construct_prompt_with_validation(
    num_samples: int,
    pre_sampled_profiles: List[Tuple[int, int, int, List[str]]],
    all_behaviors: List[str]
) -> str:
    """Construct prompt with STRICT constraints against modification"""
    
    pre_sampled_text = ""
    # Collect all pre-sampled behaviors to ensure they are in the candidate pool
    priority_behaviors = set()
    
    for idx, (coop, emo, trust, behaviors) in enumerate(pre_sampled_profiles, 1):
        behaviors_str = ", ".join([f'"{b}"' for b in behaviors])
        pre_sampled_text += f"""Profile {idx}: cooperation={coop}, emotion={emo}, trust={trust}, behaviors=[{behaviors_str}]\n"""
        for b in behaviors:
            priority_behaviors.add(b)
            
    # Construct Candidate Pool: Priority + Random
    # This ensures the model sees the original behaviors in the context
    remaining_slots = 100 - len(priority_behaviors)
    candidate_pool = list(priority_behaviors)
    if remaining_slots > 0 and len(all_behaviors) > 0:
        additional = random.sample(all_behaviors, min(len(all_behaviors), remaining_slots))
        candidate_pool.extend(additional)
    
    # Shuffle to avoid bias
    random.shuffle(candidate_pool)
    behaviors_text = "\n".join([f'- "{b}"' for b in candidate_pool])
    
    prompt = f"""You are a strict Data Validator. I have generated {num_samples} user profiles.
Your task is to ensure the 'behaviors' match the 'cooperation/emotion/trust' scores.

## STRICT RULES (Must Follow):
1. **DO NOT EDIT TEXT**: The content inside <behavior>...</behavior> MUST be an EXACT COPY of a string from the "Available Behavior Library" below.
2. **DO NOT TRANSLATE**: Output exactly as provided in the library. Do not translate to Chinese or summarize.
3. **DO NOT INVENT**: Do not create new phrases like "Friendly answer". Use the specific action like "Answer questions politely".
4. If the pre-sampled behavior is irrational, replace it with a fitting one **FROM THE LIBRARY**.

## Available Behavior Library (Select ONLY from here):
{behaviors_text}

## Pre-sampled Profiles (To be validated):
{pre_sampled_text}

## Output Format
Output {num_samples} profiles in XML. Keep parameters unchanged.

<user_profile>
    <initial_cooperation>0-4</initial_cooperation>
    <initial_emotion>0-3</initial_emotion>
    <initial_trust>0-5</initial_trust>
    <specific_behaviors>
        <behavior>EXACT_STRING_FROM_LIBRARY</behavior>
    </specific_behaviors>
</user_profile>

Start output:
<user_profile>
"""
    return prompt

# ==================== Parsing & Hard Validation ==================== #

def parse_profiles(text: str, valid_behavior_set: Set[str], all_actions_list: List[str]) -> List[Dict]:
    """
    Parse generated profiles and FORCE validate behaviors against the library.
    If LLM hallucinates a behavior not in the set, fallback to a random valid one.
    """
    profiles = []
    pattern = r'<user_profile>(.*?)</user_profile>'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        try:
            coop_match = re.search(r'<initial_cooperation>\s*(\d+)\s*</initial_cooperation>', match, re.IGNORECASE)
            emotion_match = re.search(r'<initial_emotion>\s*(\d+)\s*</initial_emotion>', match, re.IGNORECASE)
            trust_match = re.search(r'<initial_trust>\s*(\d+)\s*</initial_trust>', match, re.IGNORECASE)
            
            # Extract behaviors
            raw_behaviors = re.findall(r'<behavior>\s*(.*?)\s*</behavior>', match, re.IGNORECASE)
            
            if coop_match and emotion_match and trust_match:
                coop = int(coop_match.group(1))
                emotion = int(emotion_match.group(1))
                trust = int(trust_match.group(1))
                
                validated_behaviors = []
                for b in raw_behaviors:
                    # Clean up quotes
                    clean_b = b.strip().strip('"').strip("'")
                    
                    # 🚨 HARD VALIDATION: Check if behavior exists in library
                    if clean_b in valid_behavior_set:
                        validated_behaviors.append(clean_b)
                    else:
                        # Fallback: Pick a random valid behavior from the library
                        # This fixes the "Chinese summary" or "Hallucination" issue
                        if all_actions_list:
                            fallback = random.choice(all_actions_list)
                            validated_behaviors.append(fallback)
                
                # Ensure at least one behavior
                if not validated_behaviors and all_actions_list:
                    validated_behaviors = [random.choice(all_actions_list)]
                
                profile = {
                    'cooperation': coop,
                    'emotion': emotion,
                    'trust': trust,
                    'behaviors': validated_behaviors
                }
                profiles.append(profile)
        except Exception:
            continue
    return profiles

def smart_dedup(profiles: List[Dict], max_per_key: int = 200) -> List[Dict]:
    """Smart deduplication"""
    unique_profiles = []
    seen_exact = set()
    key_counts = Counter()
    
    for profile in profiles:
        behaviors_tuple = tuple(sorted(profile.get('behaviors', [])))
        exact_key = (
            profile.get('cooperation'),
            profile.get('emotion'),
            profile.get('trust'),
            behaviors_tuple
        )
        
        if exact_key in seen_exact:
            continue
        
        param_key = (
            profile.get('cooperation'),
            profile.get('emotion'),
            profile.get('trust')
        )
        
        if key_counts[param_key] < max_per_key:
            unique_profiles.append(profile)
            seen_exact.add(exact_key)
            key_counts[param_key] += 1
    
    return unique_profiles

# ==================== Worker Process ==================== #

def worker_process(
    worker_id: int,
    gpu_id: int,
    model_path: str,
    analysis: Dict,
    behavior_lib: Dict,
    output_file: str,
    shared_dict: dict,
    global_target: int,
    progress_queue: Queue,
    seed: int
):
    """Worker: A100 Single Card Execution"""
    llm = None
    rng = random.Random(seed + worker_id * 1000)
    
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"[Worker {worker_id}] 🚀 Starting (GPU: {gpu_id})")
        
        # Prepare Validation Set
        all_actions = behavior_lib.get('all_actions', [])
        valid_behavior_set = set(all_actions)
        
        if not all_actions:
            print(f"[Worker {worker_id}] ❌ Error: Behavior library is empty!")
            return

        from vllm import LLM, SamplingParams

        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            max_model_len=8192,
            dtype="bfloat16",
            trust_remote_code=True,
            enforce_eager=False
        )
        
        consecutive_errors = 0
        round_num = 0
        
        while shared_dict.get('deduped_count', 0) < global_target:
            round_num += 1
            if consecutive_errors > 3:
                time.sleep(5)
            
            batch_size = 70
            
            # 1. Pre-sample
            pre_sampled_profiles = []
            for _ in range(batch_size):
                profile = sample_random_profile(behavior_lib, rng, analysis)
                pre_sampled_profiles.append(profile)
            
            # 2. Construct Prompt (With strict rules)
            current_prompt = construct_prompt_with_validation(
                batch_size, 
                pre_sampled_profiles,
                all_actions
            )
            
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.95,
                max_tokens=min(batch_size * 200, 8192),
                min_tokens=100,
                n=1
            )
            
            try:
                outputs = llm.generate([current_prompt], sampling_params)
                text = outputs[0].outputs[0].text
                
                # 3. Parse & Force Validate
                profiles = parse_profiles(text, valid_behavior_set, all_actions)
                
                if profiles:
                    consecutive_errors = 0
                    
                    # Write to file
                    with open(output_file, 'a', encoding='utf-8') as f:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                        try:
                            for p in profiles:
                                f.write(json.dumps(p, ensure_ascii=False) + '\n')
                                f.flush()
                        finally:
                            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    
                    progress_queue.put(('new_data', len(profiles)))
                    print(f"[Worker {worker_id}] ✅ Round {round_num}: Wrote {len(profiles)}")
                else:
                    consecutive_errors += 1
                    print(f"[Worker {worker_id}] ⚠️ Round {round_num}: Parse empty")
                    
            except Exception as e:
                print(f"[Worker {worker_id}] ❌ Round {round_num}: Exception: {e}")
                consecutive_errors += 1
        
    except Exception as e:
        print(f"[Worker {worker_id}] ❌ Fatal Error: {e}")
        traceback.print_exc()
    finally:
        if llm: del llm
        import gc; gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except: pass

# ==================== Main Controller ==================== #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--analysis_file", type=str, required=True)
    parser.add_argument("--behavior_library", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=960)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    force_cleanup()
    
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f: pass
    
    analysis = load_analysis(args.analysis_file)
    behavior_lib = load_behavior_library(args.behavior_library)
    
    manager = Manager()
    shared_dict = manager.dict()
    shared_dict['deduped_count'] = 0
    
    gpu_allocation = list(range(min(args.num_workers, 8)))
    progress_queue = Queue()
    processes = []
    
    print(f"🚀 Starting {len(gpu_allocation)} Workers...")
    
    for i, gpu_id in enumerate(gpu_allocation):
        p = Process(
            target=worker_process,
            args=(
                i+1,
                gpu_id,
                args.model_path,
                analysis,
                behavior_lib,
                str(output_path),
                shared_dict,
                args.num_samples,
                progress_queue,
                args.seed
            )
        )
        p.start()
        processes.append(p)
        time.sleep(1)
    
    # Monitoring
    last_update_time = time.time()
    start_time = time.time()
    
    try:
        while True:
            alive_processes = [p for p in processes if p.is_alive()]
            if not alive_processes: break
            
            has_new_data = False
            try:
                while not progress_queue.empty():
                    progress_queue.get_nowait()
                    has_new_data = True
                    last_update_time = time.time()
            except: pass
            
            if has_new_data:
                # Deduplication check
                all_profiles = []
                with open(output_path, 'r') as f:
                    for line in f:
                        try: all_profiles.append(json.loads(line))
                        except: pass
                
                unique_profiles = smart_dedup(all_profiles, max_per_key=200)
                shared_dict['deduped_count'] = len(unique_profiles)
                
                print(f"\r📊 Progress: {len(unique_profiles)}/{args.num_samples}", end="", flush=True)
            
            if shared_dict.get('deduped_count', 0) >= args.num_samples:
                print("\n✅ Target reached!")
                break
                
            if time.time() - last_update_time > 600:
                print("\n❌ Timeout")
                break
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    print("\n🛑 Stopping processes...")
    for p in processes:
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
            if p.is_alive(): p.kill()
            
    force_cleanup()
    print("\n🎉 Done")

if __name__ == "__main__":
    main()