#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyze Chatbot Judgment Errors - V5 (Optimal Version)
✅ Fine-grained completion rate analysis
✅ Weight recommendations based on historical data
✅ Success rate considering dimension combinations
✅ Prohibit extreme profiles
✅ Provide verifiable adjustment suggestions
"""

import json
import re
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
import argparse
import numpy as np

# ==================== Extraction Functions ==================== #

def extract_score(content: str, tag: str) -> int:
    """Extract score from XML tag"""
    try:
        match = re.search(rf'<{tag}[^>]*>(\d+)</{tag}>', content, re.IGNORECASE)
        return int(match.group(1)) if match else None
    except:
        return None

def extract_initial_state(dialogue: Dict) -> Dict:
    """Extract initial state from ground_truth_history"""
    ground_truth = dialogue.get('ground_truth_history', [])
    
    if not ground_truth or len(ground_truth) < 2:
        return {'cooperation': 2, 'emotion': 1, 'trust': 2}
    
    first_user_msg = ground_truth[1]
    content = first_user_msg.get('content', '')
    
    coop_match = re.search(r'<user_cooperation>(\d+)</user_cooperation>', content)
    emotion_match = re.search(r'<user_emotion>(\d+)</user_emotion>', content)
    trust_match = re.search(r'<user_trust>(\d+)</user_trust>', content)
    
    return {
        'cooperation': int(coop_match.group(1)) if coop_match else 2,
        'emotion': int(emotion_match.group(1)) if emotion_match else 1,
        'trust': int(trust_match.group(1)) if trust_match else 2
    }

def calculate_completion_rate(dialogues: List[Dict]) -> float:
    """Calculate completion rate"""
    if not dialogues:
        return 0.0
    
    success_count = sum(1 for d in dialogues 
                       if d.get('batch_statuses', []) 
                       and d['batch_statuses'][-1] == 1)
    return success_count / len(dialogues)

def calculate_average_profile(dialogues: List[Dict]) -> Dict:
    """Calculate average user profile"""
    cooperation_values = []
    emotion_values = []
    trust_values = []
    
    for dialogue in dialogues:
        user_params = extract_initial_state(dialogue)
        cooperation_values.append(user_params.get('cooperation', 2))
        emotion_values.append(user_params.get('emotion', 1))
        trust_values.append(user_params.get('trust', 2))
    
    if not cooperation_values:
        return {
            'avg_cooperation': 2.0,
            'avg_emotion': 1.5,
            'avg_trust': 3.0,
            'cooperation_std': 0.0,
            'emotion_std': 0.0,
            'trust_std': 0.0
        }
    
    return {
        'avg_cooperation': float(np.mean(cooperation_values)),
        'avg_emotion': float(np.mean(emotion_values)),
        'avg_trust': float(np.mean(trust_values)),
        'cooperation_std': float(np.std(cooperation_values)),
        'emotion_std': float(np.std(emotion_values)),
        'trust_std': float(np.std(trust_values))
    }

# ==================== Core Analysis Functions ==================== #

def analyze_dimension_success_rate(dialogues: List[Dict]) -> Dict:
    """Analyze success rate for each dimension value"""
    
    cooperation_stats = defaultdict(lambda: {'success': 0, 'total': 0})
    emotion_stats = defaultdict(lambda: {'success': 0, 'total': 0})
    trust_stats = defaultdict(lambda: {'success': 0, 'total': 0})
    
    for dialogue in dialogues:
        user_params = extract_initial_state(dialogue)
        batch_statuses = dialogue.get('batch_statuses', [])
        
        if not batch_statuses:
            continue
        
        coop = user_params.get('cooperation', 2)
        emotion = user_params.get('emotion', 1)
        trust = user_params.get('trust', 2)
        
        is_success = (batch_statuses[-1] == 1)
        
        cooperation_stats[coop]['total'] += 1
        emotion_stats[emotion]['total'] += 1
        trust_stats[trust]['total'] += 1
        
        if is_success:
            cooperation_stats[coop]['success'] += 1
            emotion_stats[emotion]['success'] += 1
            trust_stats[trust]['success'] += 1
    
    def calc_stats(stats):
        return {
            k: {
                'success_rate': v['success'] / v['total'] if v['total'] > 0 else 0,
                'count': v['total']
            }
            for k, v in stats.items()
        }
    
    return {
        'cooperation': calc_stats(cooperation_stats),
        'emotion': calc_stats(emotion_stats),
        'trust': calc_stats(trust_stats)
    }

def analyze_combination_success_rate(dialogues: List[Dict], top_k: int = 20) -> Dict:
    """Analyze success rate for dimension combinations"""
    
    combination_stats = defaultdict(lambda: {'success': 0, 'total': 0})
    
    for dialogue in dialogues:
        user_params = extract_initial_state(dialogue)
        batch_statuses = dialogue.get('batch_statuses', [])
        
        if not batch_statuses:
            continue
        
        coop = user_params.get('cooperation', 2)
        emotion = user_params.get('emotion', 1)
        trust = user_params.get('trust', 2)
        
        combo_key = f"coop_{coop}_emotion_{emotion}_trust_{trust}"
        
        combination_stats[combo_key]['total'] += 1
        
        if batch_statuses[-1] == 1:
            combination_stats[combo_key]['success'] += 1
    
    # Only keep combinations with sample count >= 5
    valid_combos = {
        k: {
            'success_rate': v['success'] / v['total'],
            'count': v['total']
        }
        for k, v in combination_stats.items()
        if v['total'] >= 5
    }
    
    # Sort by success rate
    sorted_combos = sorted(valid_combos.items(), key=lambda x: x[1]['success_rate'])
    
    hardest = dict(sorted_combos[:top_k])
    easiest = dict(sorted_combos[-top_k:])
    
    return {
        'all_combinations': valid_combos,
        'hardest_combinations': hardest,
        'easiest_combinations': easiest
    }

def recommend_sampling_weights(
    completion_rate: float,
    dimension_stats: Dict,
    combination_stats: Dict,
    target_rate: float = 0.5,
    allow_extreme: bool = False
) -> Dict:
    """
    Recommend sampling weights for the next round based on current data
    
    Args:
        completion_rate: Current completion rate
        dimension_stats: Dimension statistics
        combination_stats: Combination statistics
        target_rate: Target completion rate
        allow_extreme: Whether to allow extreme profiles
    
    Returns:
        Recommended weights and adjustment reasons
    """
    
    # Define valid ranges
    if allow_extreme:
        valid_coop = list(range(5))
        valid_emotion = list(range(4))
        valid_trust = list(range(6))
    else:
        valid_coop = [1, 2, 3]
        valid_emotion = [1, 2]
        valid_trust = [1, 2, 3, 4]
    
    # Calculate deviation
    deviation = completion_rate - target_rate
    
    # Base weights
    base_coop_weights = {i: 1.0 for i in valid_coop}
    base_emotion_weights = {i: 1.0 for i in valid_emotion}
    base_trust_weights = {i: 1.0 for i in valid_trust}
    
    # Get statistics
    coop_stats = dimension_stats['cooperation']
    emotion_stats = dimension_stats['emotion']
    trust_stats = dimension_stats['trust']
    
    # Adjust weights based on deviation
    if deviation < -0.1:  # Completion rate < 40%, too difficult
        adjustment_factor = abs(deviation) * 2
        
        coop_weights_dict = {}
        for i in valid_coop:
            sr = coop_stats.get(i, {}).get('success_rate', 0.5)
            weight = base_coop_weights[i] * (1 + adjustment_factor * sr)
            coop_weights_dict[i] = weight
        
        emotion_weights_dict = {}
        for i in valid_emotion:
            sr = emotion_stats.get(i, {}).get('success_rate', 0.5)
            weight = base_emotion_weights[i] * (1 + adjustment_factor * sr)
            emotion_weights_dict[i] = weight
        
        trust_weights_dict = {}
        for i in valid_trust:
            sr = trust_stats.get(i, {}).get('success_rate', 0.5)
            weight = base_trust_weights[i] * (1 + adjustment_factor * sr)
            trust_weights_dict[i] = weight
        
        reason = f"Completion rate {completion_rate:.1%}, below target {target_rate:.1%}, increasing easier profiles (adjustment factor {adjustment_factor:.2f})"
    
    elif deviation > 0.1:  # Completion rate > 60%, too easy
        adjustment_factor = deviation * 2
        
        coop_weights_dict = {}
        for i in valid_coop:
            sr = coop_stats.get(i, {}).get('success_rate', 0.5)
            weight = base_coop_weights[i] * (1 + adjustment_factor * (1 - sr))
            coop_weights_dict[i] = weight
        
        emotion_weights_dict = {}
        for i in valid_emotion:
            sr = emotion_stats.get(i, {}).get('success_rate', 0.5)
            weight = base_emotion_weights[i] * (1 + adjustment_factor * (1 - sr))
            emotion_weights_dict[i] = weight
        
        trust_weights_dict = {}
        for i in valid_trust:
            sr = trust_stats.get(i, {}).get('success_rate', 0.5)
            weight = base_trust_weights[i] * (1 + adjustment_factor * (1 - sr))
            trust_weights_dict[i] = weight
        
        reason = f"Completion rate {completion_rate:.1%}, above target {target_rate:.1%}, increasing harder profiles (adjustment factor {adjustment_factor:.2f})"
    
    else:  # 40% <= Completion rate <= 60%, close to target
        coop_weights_dict = {}
        for i in valid_coop:
            count = coop_stats.get(i, {}).get('count', 1)
            weight = base_coop_weights[i] * (1 + 0.1 / count)
            coop_weights_dict[i] = weight
        
        emotion_weights_dict = {}
        for i in valid_emotion:
            count = emotion_stats.get(i, {}).get('count', 1)
            weight = base_emotion_weights[i] * (1 + 0.1 / count)
            emotion_weights_dict[i] = weight
        
        trust_weights_dict = {}
        for i in valid_trust:
            count = trust_stats.get(i, {}).get('count', 1)
            weight = base_trust_weights[i] * (1 + 0.1 / count)
            trust_weights_dict[i] = weight
        
        reason = f"Completion rate {completion_rate:.1%}, close to target {target_rate:.1%}, maintaining current distribution and balancing sample counts"
    
    # Normalize
    coop_sum = sum(coop_weights_dict.values())
    emotion_sum = sum(emotion_weights_dict.values())
    trust_sum = sum(trust_weights_dict.values())
    
    # Generate complete weight lists (extreme values = 0)
    coop_weights = []
    for i in range(5):
        if i in coop_weights_dict:
            coop_weights.append(coop_weights_dict[i] / coop_sum)
        else:
            coop_weights.append(0.0)
    
    emotion_weights = []
    for i in range(4):
        if i in emotion_weights_dict:
            emotion_weights.append(emotion_weights_dict[i] / emotion_sum)
        else:
            emotion_weights.append(0.0)
    
    trust_weights = []
    for i in range(6):
        if i in trust_weights_dict:
            trust_weights.append(trust_weights_dict[i] / trust_sum)
        else:
            trust_weights.append(0.0)
    
    return {
        'cooperation_weights': coop_weights,
        'emotion_weights': emotion_weights,
        'trust_weights': trust_weights,
        'adjustment_reason': reason,
        'deviation': deviation,
        'adjustment_factor': abs(deviation) * 2 if abs(deviation) > 0.1 else 0.1,
        'valid_ranges': {
            'cooperation': valid_coop,
            'emotion': valid_emotion,
            'trust': valid_trust
        }
    }

def generate_optimization_suggestions(
    completion_rate: float,
    avg_profile: Dict,
    dimension_stats: Dict,
    combination_stats: Dict,
    recommended_weights: Dict
) -> List[str]:
    """Generate optimization suggestions"""
    
    suggestions = []
    
    # 1. Completion rate analysis
    if completion_rate < 0.3:
        suggestions.append(
            f"🔴 **Extremely low completion rate** ({completion_rate:.1%}): Current test is too difficult, strongly recommend increasing easier profiles"
        )
    elif completion_rate < 0.45:
        suggestions.append(
            f"🟡 **Low completion rate** ({completion_rate:.1%}): Moderately increase easier profiles"
        )
    elif completion_rate > 0.7:
        suggestions.append(
            f"🔴 **Excessively high completion rate** ({completion_rate:.1%}): Current test is too easy, strongly recommend increasing harder profiles"
        )
    elif completion_rate > 0.55:
        suggestions.append(
            f"🟡 **High completion rate** ({completion_rate:.1%}): Moderately increase harder profiles"
        )
    else:
        suggestions.append(
            f"🟢 **Ideal completion rate** ({completion_rate:.1%}): Close to 50%, maintain current distribution"
        )
    
    # 2. Dimension analysis
    coop_stats = dimension_stats['cooperation']
    
    if coop_stats:
        weakest_coop = min(coop_stats.items(), key=lambda x: x[1]['success_rate'])
        if weakest_coop[1]['success_rate'] < 0.3:
            suggestions.append(
                f"⚠️ **Weakness**: Cooperation={weakest_coop[0]} has extremely low success rate ({weakest_coop[1]['success_rate']:.1%}), "
                f"recommend increasing such profiles for targeted training"
            )
    
    # 3. Combination analysis
    hardest = combination_stats.get('hardest_combinations', {})
    if hardest:
        hardest_combo = list(hardest.items())[0]
        suggestions.append(
            f"🎯 **Hardest combination**: {hardest_combo[0]} success rate {hardest_combo[1]['success_rate']:.1%}, "
            f"sample count {hardest_combo[1]['count']}, recommend increasing"
        )
    
    easiest = combination_stats.get('easiest_combinations', {})
    if easiest:
        easiest_combo = list(easiest.items())[-1]
        suggestions.append(
            f"✅ **Easiest combination**: {easiest_combo[0]} success rate {easiest_combo[1]['success_rate']:.1%}, "
            f"sample count {easiest_combo[1]['count']}, recommend decreasing"
        )
    
    # 4. Weight recommendation
    suggestions.append(
        f"📊 **Recommended weights**: {recommended_weights['adjustment_reason']}"
    )
    
    return suggestions

# ==================== Main Function ==================== #

def analyze_all_mistakes(rollout_file: str, output_file: str, target_rate: float = 0.5, allow_extreme: bool = False):
    """Analyze all dialogue errors and generate actionable analysis report"""
    
    print(f"📖 Loading data from {rollout_file}...")
    
    all_dialogues = []
    
    with open(rollout_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                dialogue = json.loads(line.strip())
                all_dialogues.append(dialogue)
            except:
                continue
    
    print(f"✅ Total dialogues: {len(all_dialogues)}")
    
    if not all_dialogues:
        print("❌ No dialogues found")
        return
    
    # 1. Calculate completion rate
    completion_rate = calculate_completion_rate(all_dialogues)
    
    # 2. Calculate average user profile
    avg_profile = calculate_average_profile(all_dialogues)
    
    # 3. Analyze dimension success rates
    dimension_stats = analyze_dimension_success_rate(all_dialogues)
    
    # 4. Analyze combination success rates
    combination_stats = analyze_combination_success_rate(all_dialogues)
    
    # 5. Recommend sampling weights
    recommended_weights = recommend_sampling_weights(
        completion_rate,
        dimension_stats,
        combination_stats,
        target_rate,
        allow_extreme
    )
    
    # 6. Generate optimization suggestions
    suggestions = generate_optimization_suggestions(
        completion_rate,
        avg_profile,
        dimension_stats,
        combination_stats,
        recommended_weights
    )
    
    # 7. Calculate global statistics
    all_turns = [d.get('turns', 0) for d in all_dialogues]
    avg_turns = float(np.mean(all_turns)) if all_turns else 0.0
    
    # 8. Generate analysis report
    analysis_report = {
        # Core statistics
        'total_samples': len(all_dialogues),
        'completion_rate': completion_rate,
        'target_completion_rate': target_rate,
        'deviation': completion_rate - target_rate,
        
        # Average profile
        'avg_cooperation': avg_profile['avg_cooperation'],
        'avg_emotion': avg_profile['avg_emotion'],
        'avg_trust': avg_profile['avg_trust'],
        'cooperation_std': avg_profile['cooperation_std'],
        'emotion_std': avg_profile['emotion_std'],
        'trust_std': avg_profile['trust_std'],
        'avg_turns': avg_turns,
        
        # Detailed statistics
        'dimension_stats': dimension_stats,
        'combination_stats': combination_stats,
        
        # Recommended weights
        'recommended_weights': recommended_weights,
        
        # Optimization suggestions
        'optimization_suggestions': suggestions,
    }
    
    # Save analysis report
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_report, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Analysis report saved to {output_file}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"📊 Analysis Summary")
    print(f"{'='*70}")
    print(f"Total dialogues:     {len(all_dialogues)}")
    print(f"Completion rate:     {completion_rate:.2%} (Target: {target_rate:.2%})")
    print(f"Deviation:           {completion_rate - target_rate:+.2%}")
    print(f"Avg cooperation:     {avg_profile['avg_cooperation']:.2f} ± {avg_profile['cooperation_std']:.2f}")
    print(f"Avg emotion:         {avg_profile['avg_emotion']:.2f} ± {avg_profile['emotion_std']:.2f}")
    print(f"Avg trust:           {avg_profile['avg_trust']:.2f} ± {avg_profile['trust_std']:.2f}")
    print(f"Avg turns:           {avg_turns:.2f}")
    
    print(f"\n📊 Recommended Weights:")
    print(f"Cooperation: {[f'{w:.3f}' for w in recommended_weights['cooperation_weights']]}")
    print(f"Emotion:     {[f'{w:.3f}' for w in recommended_weights['emotion_weights']]}")
    print(f"Trust:       {[f'{w:.3f}' for w in recommended_weights['trust_weights']]}")
    print(f"Reason:      {recommended_weights['adjustment_reason']}")
    
    print(f"\n🎯 Optimization Suggestions:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"   {i}. {suggestion}")
    
    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout_file", type=str, required=True, help="Rollout JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Analysis report output file")
    parser.add_argument("--target_rate", type=float, default=0.5, help="Target completion rate (default 50%%)")
    parser.add_argument("--allow_extreme", action='store_true', help="Whether to allow extreme profiles")
    args = parser.parse_args()
    
    analyze_all_mistakes(args.rollout_file, args.output_file, args.target_rate, args.allow_extreme)