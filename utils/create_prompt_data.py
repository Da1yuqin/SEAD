#!/usr/bin/env python
# create_prompt_data.py
# -*- coding: utf-8 -*-
"""
Data Generation Script - Telemarketing Scenario Version (Fixed)
✅ Remove extreme values
✅ Support specified distribution ratios
✅ Ensure reasonable distribution

For Evaluation, run:
python utils/create_prompt_data.py \
    --train_samples 0 \
    --test_samples 1000 \
    --behavior_library ./assets/client_action.jsonl \
    --out_dir ./outputs/evaluation/test_set/ \
    --temp_dir ./outputs/evaluation/test_set/user_param/
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

def read_prompt(file_path: Path) -> str:
    """Read prompt file"""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return file_path.read_text(encoding="utf-8").strip()

def load_behavior_library(library_file: Path) -> Dict:
    """Load behavior library"""
    if not library_file.exists():
        raise FileNotFoundError(f"Behavior library file not found: {library_file}")
    
    with open(library_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    categories = data.get("User_Persona_Classification", {})
    if not categories:
        raise ValueError("Behavior library format error: 'User_Persona_Classification' (User Persona Classification) key not found")
    
    all_actions = []
    for cat, acts in categories.items():
        if acts and isinstance(acts, list):
            all_actions.extend(acts)
    
    if not all_actions:
        raise ValueError("No behavior data found in the library")
    
    print(f"✅ Loaded behavior library: {len(categories)} categories, {len(all_actions)} behaviors")
    
    return {"all_actions": all_actions, "categories": categories}

def load_profiles_from_file(
    file_path: str, 
    train_samples: int, 
    test_samples: int
) -> Tuple[Optional[List[Dict]], Optional[List[Dict]]]:
    """Load profiles from file and split into train/test sets"""
    
    if not Path(file_path).exists():
        print(f"⚠️  Profile file does not exist: {file_path}")
        return None, None
    
    all_profiles = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                profile = json.loads(line)
                if 'cooperation' in profile and 'emotion' in profile and 'trust' in profile:
                    all_profiles.append(profile)
                else:
                    print(f"⚠️  Skipping malformed profile (Line {line_num}): {line[:50]}")
            except json.JSONDecodeError:
                print(f"⚠️  Skipping invalid JSON (Line {line_num}): {line[:50]}")
                continue
    
    print(f"\n✅ Loaded profiles from file: {file_path}")
    print(f"   Total: {len(all_profiles)}")
    
    total_needed = train_samples + test_samples
    if len(all_profiles) < total_needed:
        print(f"⚠️  Insufficient number of profiles! Needed {total_needed}, actual {len(all_profiles)}")
        print(f"   Will supplement with random profiles")
        return None, None
    
    train_profiles = all_profiles[:train_samples]
    test_profiles = all_profiles[train_samples:train_samples + test_samples]
    
    print(f"   Train set: {len(train_profiles)}")
    print(f"   Test set: {len(test_profiles)}")
    
    print(f"\n📊 Train set profile distribution:")
    print_profile_distribution(train_profiles)
    
    return train_profiles, test_profiles

def print_profile_distribution(profiles: List[Dict]):
    """Print profile distribution statistics"""
    coop_dist = {}
    emotion_dist = {}
    trust_dist = {}
    
    for p in profiles:
        coop = p.get('cooperation', -1)
        emotion = p.get('emotion', -1)
        trust = p.get('trust', -1)
        
        coop_dist[coop] = coop_dist.get(coop, 0) + 1
        emotion_dist[emotion] = emotion_dist.get(emotion, 0) + 1
        trust_dist[trust] = trust_dist.get(trust, 0) + 1
    
    print(f"   Cooperation distribution: {dict(sorted(coop_dist.items()))}")
    print(f"   Emotion distribution:   {dict(sorted(emotion_dist.items()))}")
    print(f"   Trust distribution: {dict(sorted(trust_dist.items()))}")
    
    total = len(profiles)
    print(f"\n   Cooperation Ratio:")
    for k in sorted(coop_dist.keys()):
        print(f"      {k}: {coop_dist[k]/total*100:.1f}%")
    
    print(f"   Emotion Ratio:")
    for k in sorted(emotion_dist.keys()):
        print(f"      {k}: {emotion_dist[k]/total*100:.1f}%")
    
    print(f"   Trust Ratio:")
    for k in sorted(trust_dist.keys()):
        print(f"      {k}: {trust_dist[k]/total*100:.1f}%")

def generate_random_profiles_with_distribution(
    num_samples: int, 
    behavior_library: Dict,
    cooperation_dist: Dict[int, float] = None,
    emotion_dist: Dict[int, float] = None,
    trust_dist: Dict[int, float] = None
) -> List[Dict]:
    """
    Generate random profiles (supports specified distribution)
    
    Args:
        num_samples: Number of samples
        behavior_library: Behavior library
        cooperation_dist: Cooperation distribution {value: ratio}, e.g., {1: 0.3, 2: 0.4, 3: 0.3}
        emotion_dist: Emotion distribution {value: ratio}
        trust_dist: Trust distribution {value: ratio}
    
    Returns:
        List[Dict]: List of profiles
    """
    all_actions = behavior_library.get('all_actions', [])
    
    # ✅ Default distribution (remove extreme values)
    if cooperation_dist is None:
        # cooperation: 1-3 (remove 0 and 4)
        cooperation_dist = {
            1: 0.30,  # 30% - Cold/Resistant
            2: 0.40,  # 40% - Hesitant
            3: 0.30,  # 30% - Willing to cooperate
        }
    
    if emotion_dist is None:
        # emotion: 1-2 (remove 0 and 3)
        emotion_dist = {
            1: 0.50,  # 50% - Slightly impatient
            2: 0.50,  # 50% - Calm/Neutral
        }
    
    if trust_dist is None:
        # trust: 1-4 (remove 0 and 5)
        trust_dist = {
            1: 0.20,  # 20% - Obviously distrustful
            2: 0.30,  # 30% - Skeptical
            3: 0.30,  # 30% - Cautious observation
            4: 0.20,  # 20% - Basic trust
        }
    
    # ✅ Validate distribution sum is 1
    def validate_dist(dist, name):
        total = sum(dist.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"{name} distribution sum must be 1.0, current is {total}")
    
    validate_dist(cooperation_dist, "cooperation")
    validate_dist(emotion_dist, "emotion")
    validate_dist(trust_dist, "trust")
    
    profiles = []
    
    # Generate count for each value based on ratio
    coop_values = []
    for value, ratio in cooperation_dist.items():
        count = int(num_samples * ratio)
        coop_values.extend([value] * count)
    
    emotion_values = []
    for value, ratio in emotion_dist.items():
        count = int(num_samples * ratio)
        emotion_values.extend([value] * count)
    
    trust_values = []
    for value, ratio in trust_dist.items():
        count = int(num_samples * ratio)
        trust_values.extend([value] * count)
    
    # ✅ Pad to num_samples (handle rounding errors)
    while len(coop_values) < num_samples:
        coop_values.append(random.choice(list(cooperation_dist.keys())))
    while len(emotion_values) < num_samples:
        emotion_values.append(random.choice(list(emotion_dist.keys())))
    while len(trust_values) < num_samples:
        trust_values.append(random.choice(list(trust_dist.keys())))
    
    # ✅ Truncate to num_samples
    coop_values = coop_values[:num_samples]
    emotion_values = emotion_values[:num_samples]
    trust_values = trust_values[:num_samples]
    
    # ✅ Shuffle order
    random.shuffle(coop_values)
    random.shuffle(emotion_values)
    random.shuffle(trust_values)
    
    # ✅ Combine into profile
    for i in range(num_samples):
        num_behaviors = random.randint(1, 3)
        selected = random.sample(all_actions, min(num_behaviors, len(all_actions))) if all_actions else []
        
        profile = {
            'cooperation': coop_values[i],
            'emotion': emotion_values[i],
            'trust': trust_values[i],
            'behaviors': selected
        }
        profiles.append(profile)
    
    return profiles

def fill_client_prompt(template: str, params: Dict) -> str:
    """Fill Client Prompt (Restaurant Owner Persona)"""
    text = template.replace("${initial_cooperation}", str(params["cooperation"]))
    text = text.replace("${initial_emotion}", str(params["emotion"]))
    text = text.replace("${initial_trust}", str(params["trust"]))
    
    behaviors = params.get("behaviors", ["No specific behavior"])
    
    if isinstance(behaviors, str):
        behaviors = [behaviors]
    elif not isinstance(behaviors, list):
        behaviors = ["No specific behavior"]
    
    behaviors = [b.strip() for b in behaviors if b and b.strip()]
    if not behaviors:
        behaviors = ["No specific behavior"]
    
    behavior_text = "\n".join([f"- {b}" for b in behaviors])
    text = text.replace("${specific_behaviors}", behavior_text)
    
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    return text

def main():
    parser = argparse.ArgumentParser(description="Generate Chatbot Training Data")
    parser.add_argument("--mode", default="chatbot", choices=["chatbot", "client"])
    parser.add_argument("--train_samples", type=int, required=True, help="Number of training samples")
    parser.add_argument("--test_samples", type=int, required=True, help="Number of test samples")
    parser.add_argument("--out_dir", default="./outputs/prompt_data")
    parser.add_argument("--txt_dir", default="./verl/trainer/config/format_prompt")
    parser.add_argument("--temp_dir", default="./outputs/temp")
    parser.add_argument("--behavior_library", type=str, default="./assets/client_action.jsonl")
    parser.add_argument("--generated_profiles", type=str, default=None, 
                        help="Profile file generated from 32B (JSONL format)")
    
    
    # ✅ New: Distribution ratio parameters
    parser.add_argument("--coop_dist", type=str, default="1:0.3,2:0.4,3:0.3",
                        help="Cooperation distribution, format: 1:0.3,2:0.4,3:0.3")
    parser.add_argument("--emotion_dist", type=str, default="1:0.5,2:0.5",
                        help="Emotion distribution, format: 1:0.5,2:0.5")
    parser.add_argument("--trust_dist", type=str, default="1:0.2,2:0.3,3:0.3,4:0.2",
                        help="Trust distribution, format: 1:0.2,2:0.3,3:0.3,4:0.2")
    
    args = parser.parse_args()

    n_train = args.train_samples
    n_test = args.test_samples
    n_total = n_train + n_test
    
    print("="*70)
    print("📊 Data Generation Configuration")
    print("="*70)
    print(f"   Mode: {args.mode}")
    print(f"   Train set: {n_train} samples")
    print(f"   Test set: {n_test} samples")
    print(f"   Total: {n_total} samples")
    print("="*70)
    print()

    def parse_dist(dist_str: str) -> Dict[int, float]:
        dist = {}
        for item in dist_str.split(','):
            k, v = item.split(':')
            dist[int(k)] = float(v)
        return dist
    
    cooperation_dist = parse_dist(args.coop_dist)
    emotion_dist = parse_dist(args.emotion_dist)
    trust_dist = parse_dist(args.trust_dist)
    
    print("📊 Specified Distributions:")
    print(f"   Cooperation: {cooperation_dist}")
    print(f"   Emotion:   {emotion_dist}")
    print(f"   Trust: {trust_dist}")
    print()

    txt_dir = Path(args.txt_dir)
    client_template = read_prompt(txt_dir / "call_client.txt")
    chatbot_template = read_prompt(txt_dir / "call_chatbot.txt")
    
    print(f"✅ Read template files")
    print(f"   Client template length: {len(client_template)} chars")
    print(f"   Chatbot template length: {len(chatbot_template)} chars")
    print()
    
    behavior_library = load_behavior_library(Path(args.behavior_library))
    
    # Prioritize using 32B generated profiles
    train_profiles = None
    test_profiles = None
    
    if args.generated_profiles and Path(args.generated_profiles).exists():
        print(f"🔄 Attempting to load from 32B generated profiles...")
        train_profiles, test_profiles = load_profiles_from_file(
            args.generated_profiles,
            n_train,
            n_test
        )
    
    # Fallback: Random generation (using specified distribution)
    if train_profiles is None or test_profiles is None:
        print(f"\n⚠️  32B profiles not found or insufficient, using random generation (specified distribution)")
        
        train_profiles = generate_random_profiles_with_distribution(
            n_train, 
            behavior_library,
            cooperation_dist,
            emotion_dist,
            trust_dist
        )
        test_profiles = generate_random_profiles_with_distribution(
            n_test, 
            behavior_library,
            cooperation_dist,
            emotion_dist,
            trust_dist
        )
        
        print(f"✅ Random generation completed")
        print(f"   Train set: {len(train_profiles)}")
        print(f"   Test set: {len(test_profiles)}")
        print()
        
        print("📊 Actual train set distribution:")
        print_profile_distribution(train_profiles)
        print()
        
        print("📊 Actual test set distribution:")
        print_profile_distribution(test_profiles)
    
    all_profiles = train_profiles + test_profiles
    
    print()
    print("📝 Filling Client Prompts (Restaurant Owner Persona)...")
    
    all_client_prompts = []
    for i, params in enumerate(all_profiles):
        try:
            filled = fill_client_prompt(client_template, params)
            all_client_prompts.append(filled)
        except Exception as e:
            print(f"⚠️  Failed to fill sample {i}: {e}")
            default_params = {
                "cooperation": 2,
                "emotion": 1,
                "trust": 2,
                "behaviors": ["No specific behavior"]
            }
            filled = fill_client_prompt(client_template, default_params)
            all_client_prompts.append(filled)
    
    print(f"✅ Filling completed")
    print()
    
    temp_dir = Path(args.temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving data to {temp_dir}...")
    
    # Train set
    with open(temp_dir / "train_client_prompts.jsonl", 'w', encoding='utf-8') as f:
        for prompt in all_client_prompts[:n_train]:
            f.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + '\n')
    
    with open(temp_dir / "train_user_params.jsonl", 'w', encoding='utf-8') as f:
        for params in train_profiles:
            f.write(json.dumps(params, ensure_ascii=False) + '\n')
    
    # Test set
    with open(temp_dir / "test_client_prompts.jsonl", 'w', encoding='utf-8') as f:
        for prompt in all_client_prompts[n_train:]:
            f.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + '\n')
    
    with open(temp_dir / "test_user_params.jsonl", 'w', encoding='utf-8') as f:
        for params in test_profiles:
            f.write(json.dumps(params, ensure_ascii=False) + '\n')
    
    # Reset index counters
    with open(temp_dir / "train_index.txt", 'w') as f:
        f.write("0")
    
    with open(temp_dir / "test_index.txt", 'w') as f:
        f.write("0")
    
    print(f"✅ Save completed")
    print(f"   train_client_prompts.jsonl: {n_train} lines")
    print(f"   train_user_params.jsonl: {n_train} lines")
    print(f"   test_client_prompts.jsonl: {n_test} lines")
    print(f"   test_user_params.jsonl: {n_test} lines")
    print()
    
    print("📝 Generating Chatbot Prompts (keeping placeholders)...")
    
    chatbot_prompt = chatbot_template
    
    train_df = pd.DataFrame({
        "prompt": [[{"content": chatbot_prompt, "role": "user"}]] * n_train,
        "role": ["chatbot"] * n_train
    })
    
    test_df = pd.DataFrame({
        "prompt": [[{"content": chatbot_prompt, "role": "user"}]] * n_test,
        "role": ["val_chatbot"] * n_test
    })
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_parquet(out_dir / "train_chatbot.parquet", index=False)
    test_df.to_parquet(out_dir / "test_chatbot.parquet", index=False)
    
    print(f"✅ Saving parquet to {out_dir}/")
    print(f"   train_chatbot.parquet: {len(train_df)} rows")
    print(f"   test_chatbot.parquet: {len(test_df)} rows")
    print()
    
    print()
    print("="*70)
    print("📋 Data Example")
    print("="*70)
    print()
    print("User Params Example:")
    print("-"*70)
    print(json.dumps(all_profiles[0], ensure_ascii=False, indent=2))
    print()
    
    print("="*70)
    print("✅ Data generation completed!")
    print("="*70)

if __name__ == "__main__":
    main()