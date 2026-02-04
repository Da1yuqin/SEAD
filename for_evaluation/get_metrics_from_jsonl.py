#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dialogue Evaluation Metrics Calculation Script

This script calculates various metrics for dialogue evaluation:
- Completion Rate (CR)
- False Positive Rate (FPR)
- Stage Completion Rate (SCR)
- User Portrait Accuracy (UPA)
- State Improvements (EI/TI/CI)
- Average Turns to Success (ATT) - successful dialogues only
- Average Dialogue Turns (ADT) - all dialogues

Usage:
    python dialogue_metrics.py

Input Format:
    JSONL file where each line contains a dialogue record with the following structure:
    {
        "ground_truth_history": [
            {"role": "assistant", "content": "<stage>0</stage><cooperation_score>2</cooperation_score>..."},
            {"role": "user", "content": "<user_cooperation>2</user_cooperation><user_emotion>3</user_emotion>..."}
        ],
        "chatbot_status": 1,  # Model's prediction of success (1=success, 0=failure)
        "final_status": 1,    # Ground truth user agreement (1=agreed, 0=not agreed)
        "total_turns": 5
    }

Example:
    model_files = {
        'GPT-4': './data/gpt4_dialogues.jsonl',
        'Model-A': './data/model_a_dialogues.jsonl',
    }
    all_results = batch_calculate_all_metrics(model_files)
    generate_comparison_table(all_results)
    generate_full_latex_table(all_results)
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import re

# ==================== Helper Functions ==================== #

def extract_tag_value(text: str, tag_name: str, cast_type=None):
    """Extract value from XML-style tags"""
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


def extract_user_final_status(dialogue_item):
    """Extract user's final status from dialogue"""
    ground_truth_history = dialogue_item.get('ground_truth_history', [])
    
    if not ground_truth_history:
        return None
    
    for msg in reversed(ground_truth_history):
        if msg['role'] == 'user':
            content = msg['content']
            match = re.search(r'<user_status>\s*(-?\d+)\s*</user_status>', content, re.IGNORECASE)
            if match:
                return int(match.group(1))
            break
    
    return None


def extract_stages_from_dialogue(dialogue_item):
    """
    Extract all stages that appear in the dialogue
    Returns: List[int]
    """
    ground_truth_history = dialogue_item.get('ground_truth_history', [])
    stages = []
    
    for msg in ground_truth_history:
        if msg['role'] == 'assistant':
            content = msg['content']
            stage_val = extract_tag_value(content, 'stage', int)
            if stage_val is not None:
                stages.append(stage_val)
    return stages


def check_stage_completeness(stages: list) -> bool:
    """
    Check stage completeness:
    Logic: Find the last stage and check if all prerequisite stages were experienced before it.
    """
    if not stages:
        return False
    
    last_stage = stages[-1]
    
    if last_stage == 0:
        return True
    
    required_stages = set(range(last_stage))
    appeared_stages = set(stages)
    
    return required_stages.issubset(appeared_stages)


def extract_state_from_dialogue(dialogue_item):
    """Extract initial and final states from dialogue history"""
    ground_truth_history = dialogue_item.get('ground_truth_history', [])
    
    if not ground_truth_history:
        return None
    
    first_assistant_msg = None
    for msg in ground_truth_history:
        if msg['role'] == 'assistant':
            first_assistant_msg = msg['content']
            break
    
    last_assistant_msg = None
    for msg in reversed(ground_truth_history):
        if msg['role'] == 'assistant':
            last_assistant_msg = msg['content']
            break
    
    if not first_assistant_msg or not last_assistant_msg:
        return None
    
    def parse_state(content):
        cooperation = extract_tag_value(content, 'cooperation_score', int)
        emotion = extract_tag_value(content, 'emotion_score', int)
        trust = extract_tag_value(content, 'trust_score', int)
        return {
            'cooperation': cooperation, 'emotion': emotion, 'trust': trust
        }
    
    initial_state = parse_state(first_assistant_msg)
    final_state = parse_state(last_assistant_msg)
    
    if None in initial_state.values() or None in final_state.values():
        return None
    
    return {
        'cooperation_improvement': final_state['cooperation'] - initial_state['cooperation'],
        'emotion_improvement': final_state['emotion'] - initial_state['emotion'],
        'trust_improvement': final_state['trust'] - initial_state['trust']
    }


# ==================== UPA Calculation Functions ==================== #

def calculate_upa_for_dialogue(dialogue_item):
    """Calculate User Portrait Accuracy (UPA) for a single dialogue"""
    ground_truth_history = dialogue_item.get('ground_truth_history', [])
    if not ground_truth_history: return None
    
    cooperation_errors, emotion_errors, trust_errors = [], [], []
    
    for i in range(len(ground_truth_history) - 1):
        if ground_truth_history[i]['role'] == 'assistant':
            assistant_content = ground_truth_history[i]['content']
            pred_coop = extract_tag_value(assistant_content, 'cooperation_score', int)
            pred_emo = extract_tag_value(assistant_content, 'emotion_score', int)
            pred_trust = extract_tag_value(assistant_content, 'trust_score', int)
            
            if i + 1 < len(ground_truth_history) and ground_truth_history[i + 1]['role'] == 'user':
                user_content = ground_truth_history[i + 1]['content']
                true_coop = extract_tag_value(user_content, 'user_cooperation', int)
                true_emo = extract_tag_value(user_content, 'user_emotion', int)
                true_trust = extract_tag_value(user_content, 'user_trust', int)
                
                if all(v is not None for v in [pred_coop, pred_emo, pred_trust, true_coop, true_emo, true_trust]):
                    cooperation_errors.append(abs(pred_coop - true_coop))
                    emotion_errors.append(abs(pred_emo - true_emo))
                    trust_errors.append(abs(pred_trust - true_trust))
    
    if not cooperation_errors: return None
    
    coop_mae = np.mean(cooperation_errors)
    emo_mae = np.mean(emotion_errors)
    trust_mae = np.mean(trust_errors)
    overall_mae = np.mean([coop_mae, emo_mae, trust_mae])
    
    max_possible_error = 4.0
    upa_score = 1 - (overall_mae / max_possible_error)
    
    return {
        'upa_score': upa_score,
        'cooperation_mae': coop_mae,
        'emotion_mae': emo_mae,
        'trust_mae': trust_mae,
        'overall_mae': overall_mae,
        'valid_turns': len(cooperation_errors)
    }


# ==================== Main Calculation Functions ==================== #

def calculate_metrics_from_jsonl(jsonl_file_path):
    """Calculate all metrics from JSONL file"""
    
    all_data = {
        'cooperation_improvements': [],
        'emotion_improvements': [],
        'trust_improvements': [],
        'turns': [],
        'upa_scores': [],
        'cooperation_maes': [],
        'emotion_maes': [],
        'trust_maes': [],
        'overall_maes': []
    }
    
    total_count = 0
    true_success_count = 0
    model_success_count = 0
    false_positive_count = 0
    stage_complete_count = 0
    valid_state_count = 0
    valid_upa_count = 0
    
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue
                
                try:
                    data = json.loads(line)
                    total_count += 1
                    
                    model_final_status = data.get('chatbot_status')
                    if model_final_status == 1:
                        model_success_count += 1
                    
                    user_final_status = data.get('final_status')
                    if user_final_status == 1:
                        true_success_count += 1
                    
                    if model_final_status == 1 and user_final_status != 1:
                        false_positive_count += 1
                    
                    turns = data.get('total_turns', 0)
                    all_data['turns'].append(turns)
                    
                    stages = extract_stages_from_dialogue(data)
                    if check_stage_completeness(stages):
                        stage_complete_count += 1
                    
                    state_data = extract_state_from_dialogue(data)
                    if state_data:
                        valid_state_count += 1
                        all_data['cooperation_improvements'].append(state_data['cooperation_improvement'])
                        all_data['emotion_improvements'].append(state_data['emotion_improvement'])
                        all_data['trust_improvements'].append(state_data['trust_improvement'])
                    
                    upa_data = calculate_upa_for_dialogue(data)
                    if upa_data:
                        valid_upa_count += 1
                        all_data['upa_scores'].append(upa_data['upa_score'])
                        all_data['cooperation_maes'].append(upa_data['cooperation_mae'])
                        all_data['emotion_maes'].append(upa_data['emotion_mae'])
                        all_data['trust_maes'].append(upa_data['trust_mae'])
                        all_data['overall_maes'].append(upa_data['overall_mae'])
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_num}: {e}")
                    continue
        
        true_success_turns = []
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    user_final_status = data.get('final_status')
                    if user_final_status == 1:
                        true_success_turns.append(data.get('total_turns', 0))
                except: continue
        
        results = {
            'file': str(jsonl_file_path),
            'total_samples': total_count,
            'true_success_count': true_success_count,
            'true_completion_rate': (true_success_count / total_count * 100) if total_count > 0 else 0,
            'model_success_count': model_success_count,
            'model_completion_rate': (model_success_count / total_count * 100) if total_count > 0 else 0,
            'false_positive_count': false_positive_count,
            'false_positive_rate': (false_positive_count / model_success_count * 100) if model_success_count > 0 else 0,
            'stage_complete_count': stage_complete_count,
            'stage_completion_rate': (stage_complete_count / total_count * 100) if total_count > 0 else 0,
            'upa_mean': np.mean(all_data['upa_scores']) if all_data['upa_scores'] else None,
            'upa_std': np.std(all_data['upa_scores']) if all_data['upa_scores'] else None,
            'cooperation_mae_mean': np.mean(all_data['cooperation_maes']) if all_data['cooperation_maes'] else None,
            'emotion_mae_mean': np.mean(all_data['emotion_maes']) if all_data['emotion_maes'] else None,
            'trust_mae_mean': np.mean(all_data['trust_maes']) if all_data['trust_maes'] else None,
            'overall_mae_mean': np.mean(all_data['overall_maes']) if all_data['overall_maes'] else None,
            'valid_upa_count': valid_upa_count,
            'ei_mean': np.mean(all_data['emotion_improvements']) if all_data['emotion_improvements'] else None,
            'ei_std': np.std(all_data['emotion_improvements']) if all_data['emotion_improvements'] else None,
            'ti_mean': np.mean(all_data['trust_improvements']) if all_data['trust_improvements'] else None,
            'ti_std': np.std(all_data['trust_improvements']) if all_data['trust_improvements'] else None,
            'ci_mean': np.mean(all_data['cooperation_improvements']) if all_data['cooperation_improvements'] else None,
            'ci_std': np.std(all_data['cooperation_improvements']) if all_data['cooperation_improvements'] else None,
            'att_mean': np.mean(true_success_turns) if true_success_turns else None,
            'att_std': np.std(true_success_turns) if true_success_turns else None,
            'adt_mean': np.mean(all_data['turns']) if all_data['turns'] else None,
            'adt_std': np.std(all_data['turns']) if all_data['turns'] else None,
            'adt_min': np.min(all_data['turns']) if all_data['turns'] else None,
            'adt_max': np.max(all_data['turns']) if all_data['turns'] else None,
        }
        
        return results
    
    except FileNotFoundError:
        print(f"Error: File not found - {jsonl_file_path}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ==================== Print Functions ==================== #

def print_detailed_results(result):
    """Print detailed results"""
    print(f"\n{'='*60}")
    print(f"File: {result['file']}")
    print(f"{'='*60}")
    
    print(f"\n📊 Basic Metrics:")
    print(f"  Total Samples: {result['total_samples']}")
    
    print(f"\n✅ True Completion Rate (User Actually Agreed):")
    print(f"  True Success Count: {result['true_success_count']}")
    print(f"  True Completion Rate: {result['true_completion_rate']:.2f}%")
    
    print(f"\n🔄 Stage Completion Rate (Process Completeness):")
    print(f"  Stage Complete Count: {result['stage_complete_count']}")
    print(f"  SCR: {result['stage_completion_rate']:.2f}%")
    
    print(f"\n🤖 Model Completion Rate (Model Predicted Success):")
    print(f"  Model Success Count: {result['model_success_count']}")
    print(f"  Model Completion Rate: {result['model_completion_rate']:.2f}%")
    
    print(f"\n❌ False Positive Rate (Incorrect Termination):")
    print(f"  False Positive Count: {result['false_positive_count']}")
    print(f"  False Positive Rate: {result['false_positive_rate']:.2f}%")
    
    print(f"\n🎯 User Portrait Accuracy:")
    if result['upa_mean'] is not None:
        print(f"  UPA: {result['upa_mean']:.3f} ± {result['upa_std']:.3f}")
        print(f"  Valid UPA Samples: {result['valid_upa_count']}")
    else:
        print(f"  UPA: N/A")
    
    print(f"\n⏱️ Dialogue Length:")
    if result['att_mean'] is not None:
        print(f"  ATT (Success only): {result['att_mean']:.2f} ± {result['att_std']:.2f}")
    else:
        print(f"  ATT: N/A")
    
    if result['adt_mean'] is not None:
        print(f"  ADT (All dialogues): {result['adt_mean']:.2f} ± {result['adt_std']:.2f}")
        print(f"  ADT Range: [{result['adt_min']:.0f}, {result['adt_max']:.0f}]")
    else:
        print(f"  ADT: N/A")
    
    print(f"\n📈 State Improvements:")
    if result['ei_mean'] is not None:
        print(f"  EI: {result['ei_mean']:.2f} ± {result['ei_std']:.2f}")
    if result['ti_mean'] is not None:
        print(f"  TI: {result['ti_mean']:.2f} ± {result['ti_std']:.2f}")
    if result['ci_mean'] is not None:
        print(f"  CI: {result['ci_mean']:.2f} ± {result['ci_std']:.2f}")


def generate_latex_table_row(model_name, result):
    """Generate a LaTeX table row"""
    cr = result['true_completion_rate']
    fpr = result['false_positive_rate']
    scr = result['stage_completion_rate']
    
    att = result['att_mean'] if result['att_mean'] is not None else '--'
    att_std = result['att_std'] if result['att_std'] is not None else 0
    
    adt = result['adt_mean'] if result['adt_mean'] is not None else '--'
    adt_std = result['adt_std'] if result['adt_std'] is not None else 0
    
    upa = result['upa_mean'] if result['upa_mean'] is not None else '--'
    upa_std = result['upa_std'] if result['upa_std'] is not None else 0
    
    ei = result['ei_mean'] if result['ei_mean'] is not None else '--'
    ei_std = result['ei_std'] if result['ei_std'] is not None else 0
    
    ti = result['ti_mean'] if result['ti_mean'] is not None else '--'
    ti_std = result['ti_std'] if result['ti_std'] is not None else 0
    
    ci = result['ci_mean'] if result['ci_mean'] is not None else '--'
    ci_std = result['ci_std'] if result['ci_std'] is not None else 0
    
    att_str = f"{att:.1f}$^{{\\pm {att_std:.2f}}}$" if att != '--' else "--"
    adt_str = f"{adt:.1f}$^{{\\pm {adt_std:.2f}}}$" if adt != '--' else "--"
    upa_str = f"{upa:.3f}$^{{\\pm {upa_std:.3f}}}$" if upa != '--' else "--"
    ei_str = f"{ei:.2f}$^{{\\pm {ei_std:.2f}}}$" if ei != '--' else "--"
    ti_str = f"{ti:.2f}$^{{\\pm {ti_std:.2f}}}$" if ti != '--' else "--"
    ci_str = f"{ci:.2f}$^{{\\pm {ci_std:.2f}}}$" if ci != '--' else "--"
    
    return f"{model_name} & {cr:.1f} & {fpr:.1f} & {scr:.1f} & {att_str} & {adt_str} & {upa_str} & {ei_str} & {ti_str} & {ci_str} \\\\"


def generate_full_latex_table(all_results):
    """Generate complete LaTeX table"""
    print("\n" + "="*60)
    print("LaTeX Table Rows:")
    print("="*60)
    print("Model & CR (\\%) & FPR (\\%) & SCR (\\%) & ATT & ADT & UPA & EI & TI & CI \\\\")
    print("\\hline")
    
    for model_name, result in all_results.items():
        row = generate_latex_table_row(model_name, result)
        print(row)


def generate_comparison_table(all_results):
    """Generate comparison table"""
    print("\n" + "="*60)
    print("Comparison Table:")
    print("="*60)
    
    print(f"{'Model':<25} | {'True CR':<10} | {'FPR':<10} | {'SCR':<10} | {'ATT':<10} | {'ADT':<10} | {'UPA':<10}")
    print("-" * 100)
    
    for model_name, result in all_results.items():
        true_cr = result['true_completion_rate']
        fpr = result['false_positive_rate']
        scr = result['stage_completion_rate']
        att = result['att_mean'] if result['att_mean'] is not None else 0
        adt = result['adt_mean'] if result['adt_mean'] is not None else 0
        upa = result['upa_mean'] if result['upa_mean'] is not None else 0
        
        print(f"{model_name:<25} | {true_cr:>9.2f}% | {fpr:>9.2f}% | {scr:>9.2f}% | {att:>9.2f} | {adt:>9.2f} | {upa:>9.3f}")


# ============ Usage Example ============

def batch_calculate_all_metrics(model_files):
    """
    Batch calculate metrics for multiple models
    
    Args:
        model_files: Dictionary mapping model names to JSONL file paths
    
    Returns:
        Dictionary of results for each model
    """
    all_results = {}
    for model_name, file_path in model_files.items():
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        print(f"{'='*60}")
        result = calculate_metrics_from_jsonl(file_path)
        if result:
            all_results[model_name] = result
            print_detailed_results(result)
    return all_results


if __name__ == "__main__":
    # Example usage: Replace with your actual file paths
    model_files = {
        'Model-A': './outputs/evaluation/model_a_dialogues.jsonl',
        'Model-B': './outputs/evaluation/model_b_dialogues.jsonl',
        'GPT-4': './outputs/evaluation/gpt4_dialogues.jsonl',
    }
    
    all_results = batch_calculate_all_metrics(model_files)
    generate_comparison_table(all_results)
    generate_full_latex_table(all_results)