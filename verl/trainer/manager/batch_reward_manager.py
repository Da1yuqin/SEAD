import torch
from typing import Dict, List, Any, Callable, Tuple
from collections import defaultdict
from verl.utils.reward_score import role_reward


def _select_rm_score_fn(role):
    """Select the corresponding reward calculation function"""
    if role in ['chatbot', 'call_chatbot']:
        return role_reward.compute_score_chatbot
    elif role in ['client', 'call_client']:
        return role_reward.compute_score_client
    elif role in ['val_chatbot']:
        return role_reward.compute_metric_chatbot
    elif role in ['val_client']:
        return role_reward.compute_metric_client
    else:
        raise NotImplementedError(f"Unknown data source: {role}")


class RewardManager():
    """Improved reward manager - supports batch processing of trajectories"""

    def __init__(self, tokenizer, num_examine: int, batch_size: int = 252) -> None:
        """
        Args:
            tokenizer: Tokenizer
            num_examine: Number of samples to print to console
            batch_size: Number of samples to process in one batch
        """
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.batch_size = batch_size
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'total_batches': 0,
            'role': defaultdict(int)
        }

    def _process_batch(self, data: 'DataProto', batch_indices: List[int], 
                       role: str) -> Tuple[Dict[int, float], List[List[Dict]], Dict[int, int]]:
        """
        Process a batch of data
        
        Args:
            data: DataProto object
            batch_indices: List of indices for this batch
            role: Data source type
        
        Returns:
            (reward_dict, ground_truth_histories, trajectory_index_map)
        """
        print(f"\n  [Batch Processing] Processing {len(batch_indices)} trajectories, data source: {role}")
        
        # ========== 1. Extract ground_truth_history ==========
        batch_meta_info = getattr(data, 'meta_info', {}) or {}
        all_ground_truth_histories = batch_meta_info.get('ground_truth_history', [])
        
        if not all_ground_truth_histories:
            print(f"    Warning: No ground_truth_history in meta_info")
            print(f"    meta_info keys: {list(batch_meta_info.keys())}")
            
            # Fallback: Try using dialogue_history
            all_ground_truth_histories = batch_meta_info.get('dialogue_history', [])
            if not all_ground_truth_histories:
                print(f"    Warning: No dialogue_history in meta_info either")
                return {}, [], {}
        
        # ========== 2. Collect ground_truth_histories for this batch ==========
        batch_ground_truth_histories = []
        trajectory_index_map = {}
        
        for batch_pos, orig_idx in enumerate(batch_indices):
            if orig_idx < len(all_ground_truth_histories):
                ground_truth_history = all_ground_truth_histories[orig_idx]
            else:
                ground_truth_history = []
                print(f"    Warning: Index {orig_idx} out of range (length: {len(all_ground_truth_histories)})")
            
            batch_ground_truth_histories.append(ground_truth_history)
            trajectory_index_map[batch_pos] = orig_idx
        
        # ========== 3. Get computation function ==========
        try:
            compute_score_fn = _select_rm_score_fn(role)
        except NotImplementedError as e:
            print(f"    Error: {e}")
            return {}, batch_ground_truth_histories, trajectory_index_map
        
        # ========== 4. Call batch computation function ==========
        try:
            # Decide parameters based on role type
            if role in ['val_chatbot']:
                # Test mode: pass file_path
                results = compute_score_fn(
                    ground_truth_histories=batch_ground_truth_histories,
                    meta_info=batch_meta_info,
                    file_path="./outputs/temp_dialog/test_chatbot_results.json"
                )
            elif role in ['val_client']:
                # Test mode: pass file_path
                results = compute_score_fn(
                    ground_truth_histories=batch_ground_truth_histories,
                    meta_info=batch_meta_info,
                    file_path="./outputs/temp_dialog/test_client_results.json"
                )
            else:
                # Training mode: don't pass file_path (use default path)
                results = compute_score_fn(
                    ground_truth_histories=batch_ground_truth_histories,
                    meta_info=batch_meta_info
                )
            
            # ========== 5. Defensive checks ==========
            if not isinstance(results, dict):
                print(f"    Error: compute_score_fn returned non-dict type: {type(results)}")
                return {}, batch_ground_truth_histories, trajectory_index_map
            
            # For test mode, return format is different
            if role in ['val_chatbot', 'val_client']:
                # Test mode return format: {'step': ..., 'num_trajectories': ..., 'metrics': {...}}
                # Test mode doesn't need to return trajectory-level rewards, just return empty dict
                print(f"    Test mode completed, not returning trajectory-level rewards")
                return {}, batch_ground_truth_histories, trajectory_index_map
            
            # Training mode: check summary key
            if 'summary' not in results:
                print(f"    Error: No 'summary' key in results")
                print(f"    results keys: {list(results.keys())}")
                return {}, batch_ground_truth_histories, trajectory_index_map
            
            summary = results['summary']
            
            if not isinstance(summary, dict):
                print(f"    Error: summary is not dict type: {type(summary)}")
                return {}, batch_ground_truth_histories, trajectory_index_map
            
            if 'trajectory_rewards' not in summary:
                print(f"    Error: No 'trajectory_rewards' key in summary")
                print(f"    summary keys: {list(summary.keys())}")
                return {}, batch_ground_truth_histories, trajectory_index_map
            
            trajectory_rewards_list = summary['trajectory_rewards']
            
            if not isinstance(trajectory_rewards_list, list):
                print(f"    Error: trajectory_rewards is not list type: {type(trajectory_rewards_list)}")
                return {}, batch_ground_truth_histories, trajectory_index_map
            
            # ========== 6. Extract trajectory-level rewards ==========
            batch_rewards = {}
            for batch_pos, orig_idx in trajectory_index_map.items():
                if batch_pos < len(trajectory_rewards_list):
                    traj_reward_info = trajectory_rewards_list[batch_pos]
                    if isinstance(traj_reward_info, dict) and 'total_reward' in traj_reward_info:
                        reward = float(traj_reward_info['total_reward'])
                    else:
                        reward = 0.0
                        print(f"    Warning: Reward info format error for trajectory {batch_pos}")
                else:
                    reward = 0.0
                    print(f"    Warning: No reward info for trajectory {batch_pos}")
                
                batch_rewards[orig_idx] = reward
            
            # ========== 7. Print batch statistics ==========
            batch_completion_rate = summary.get('completion_rate', 0.0)
            completed_count = summary.get('completed_count', 0)
            num_trajectories = summary.get('num_trajectories', 0)
            
            print(f"    Batch completion rate: {batch_completion_rate:.2%} ({completed_count}/{num_trajectories})")
            if batch_rewards:
                print(f"    Average reward: {sum(batch_rewards.values()) / len(batch_rewards):.4f}")
            
            return batch_rewards, batch_ground_truth_histories, trajectory_index_map
        
        except Exception as e:
            print(f"    Batch processing failed: {e}")
            import traceback
            traceback.print_exc()
            # Return default rewards
            batch_rewards = {orig_idx: 0.0 for orig_idx in trajectory_index_map.values()}
            return batch_rewards, batch_ground_truth_histories, trajectory_index_map

    def __call__(self, data: 'DataProto') -> torch.Tensor:
        """
        Calculate reward tensor, supports batch processing
        
        Args:
            data: DataProto object containing batch data
        
        Returns:
            Reward tensor with shape (batch_size, seq_len)
        """
        print(f"\n{'='*80}")
        print(f"[RewardManager] Starting data processing, total samples: {len(data)}")
        print(f"{'='*80}")
        
        # If rm_scores already exists, return directly
        if 'rm_scores' in data.batch.keys():
            print("[RewardManager] Detected rm_scores, returning directly")
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        # ============ Step 1: Group by data source ============
        print("\n[Step 1] Grouping by data source...")
        
        role_indices = defaultdict(list)  # {role: [indices]}
        valid_response_lengths = {}  # {index: valid_response_length}
        
        for i in range(len(data)):
            data_item = data[i]
            role = data_item.non_tensor_batch.get('role', 'unknown')
            role_indices[role].append(i)
            
            # Record valid response length
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_lengths[i] = valid_response_length
        
        print(f"Grouping completed, {len(role_indices)} roles in total")
        for ds, indices in role_indices.items():
            print(f"  - {ds}: {len(indices)} samples")
        
        # ============ Step 2: Process by data source in batches ============
        print(f"\n[Step 2] Processing by data source in batches (batch size: {self.batch_size})...")
        
        all_rewards = {}  # {index: reward}
        all_ground_truth_histories_for_print = {}  # {index: ground_truth_history}
        
        for role, indices in role_indices.items():
            print(f"\n  [Processing data source: {role}]")
            print(f"  Total samples: {len(indices)}")
            
            # Split into batches
            batches = [indices[i:i+self.batch_size] for i in range(0, len(indices), self.batch_size)]
            print(f"  Split into {len(batches)} batches\n")
            
            for batch_idx, batch_indices in enumerate(batches):
                print(f"  --- Batch {batch_idx + 1}/{len(batches)} ---")
                
                try:
                    batch_rewards, batch_ground_truth_histories, trajectory_index_map = self._process_batch(
                        data=data,
                        batch_indices=batch_indices,
                        role=role
                    )
                    
                    # For test mode, batch_rewards is empty, set all rewards to 0
                    if role in ['val_chatbot', 'val_client']:
                        for orig_idx in batch_indices:
                            all_rewards[orig_idx] = 0.0
                    else:
                        # Training mode: merge rewards
                        all_rewards.update(batch_rewards)
                    
                    # Save ground_truth_history for printing
                    for batch_pos, orig_idx in trajectory_index_map.items():
                        if batch_pos < len(batch_ground_truth_histories):
                            all_ground_truth_histories_for_print[orig_idx] = batch_ground_truth_histories[batch_pos]
                    
                    self.stats['total_batches'] += 1
                    self.stats['total_processed'] += len(batch_indices)
                    self.stats['role'][role] += len(batch_indices)
                    
                except Exception as e:
                    print(f"    Batch processing exception: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fill 0 rewards for failed batch
                    for orig_idx in batch_indices:
                        all_rewards[orig_idx] = 0.0
        
        # ============ Step 3: Map rewards back to tensor ============
        print(f"\n[Step 3] Mapping rewards back to tensor...")
        
        for idx, reward in all_rewards.items():
            if idx in valid_response_lengths:
                valid_response_length = valid_response_lengths[idx]
                if valid_response_length > 0:
                    reward_tensor[idx, valid_response_length - 1] = reward
            else:
                print(f"    Warning: Index {idx} has no valid response length")
        
        print(f"Mapping completed")
        
        # ============ Step 4: Print example trajectories ============
        print(f"\n[Step 4] Printing example trajectories...")
        
        for role in role_indices.keys():
            indices = role_indices[role]
            print(f"\n  [Data source: {role}]")
            
            count = 0
            for idx in indices:
                if count >= self.num_examine:
                    break
                
                if idx in all_ground_truth_histories_for_print:
                    count += 1
                    ground_truth_history = all_ground_truth_histories_for_print[idx]
                    reward = all_rewards.get(idx, 0.0)
                    
                    print(f"\n    --- Example {count} ---")
                    print(f"    Number of turns: {len(ground_truth_history)}")
                    
                    # Print first 2 turns
                    for turn_idx, turn in enumerate(ground_truth_history[:2]):
                        role_name = turn.get('role', 'unknown')
                        content = turn.get('content', '')
                        print(f"      Turn {turn_idx + 1} ({role_name}): {content[:100]}...")
                    
                    if len(ground_truth_history) > 2:
                        print(f"      ... ({len(ground_truth_history) - 2} more turns)")
                    
                    print(f"    Reward: {reward:.4f}")
        
        # ============ Print statistics ============
        print(f"\n{'='*80}")
        print("[RewardManager] Processing Statistics")
        print(f"{'='*80}")
        print(f"Total processed samples: {self.stats['total_processed']}")
        print(f"Total batches: {self.stats['total_batches']}")
        print(f"Data source distribution: {dict(self.stats['role'])}")
        
        # Safe statistics calculation
        if reward_tensor.numel() > 0:
            non_zero_mask = reward_tensor != 0
            non_zero_count = non_zero_mask.sum().item()
            
            if non_zero_count > 0:
                # Calculate statistics only on non-zero elements
                non_zero_rewards = reward_tensor[non_zero_mask]
                print(f"Average reward: {torch.mean(non_zero_rewards).item():.4f}")
                print(f"Maximum reward: {torch.max(non_zero_rewards).item():.4f}")
                print(f"Minimum reward: {torch.min(non_zero_rewards).item():.4f}")
                print(f"Non-zero rewards: {non_zero_count}")
            else:
                # All rewards are 0
                print(f"Average reward: 0.0000 (all zeros)")
                print(f"Maximum reward: 0.0000")
                print(f"Minimum reward: 0.0000")
                print(f"Non-zero rewards: 0")
        else:
            # Tensor is empty
            print(f"Warning: reward_tensor is empty")
            print(f"Average reward: N/A")
            print(f"Maximum reward: N/A")
            print(f"Minimum reward: N/A")
            print(f"Non-zero rewards: 0")
        
        print(f"{'='*80}\n")
        
        return reward_tensor

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics"""
        return {
            'total_processed': self.stats['total_processed'],
            'total_batches': self.stats['total_batches'],
            'role': dict(self.stats['role'])
        }

    def reset_statistics(self):
        """Reset statistics"""
        self.stats = {
            'total_processed': 0,
            'total_batches': 0,
            'role': defaultdict(int)
        }