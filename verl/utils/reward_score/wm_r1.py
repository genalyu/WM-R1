
import re
from typing import Dict, List
import math

def compute_wm_r1_reward(
    is_success: bool,
    traj_len: int,
    n_wm: int,
    n_wm_max: int,
    max_steps: int,
    avg_len_ref: float,
    current_episode: int,
    total_episodes: int,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.5
) -> float:
    # 1. Success Reward
    r_success = 1.0 if is_success else 0.0

    # 2. Length Reward
    p = current_episode / total_episodes
    l_budget = p * avg_len_ref + (1 - p) * max_steps
    
    # Avoid division by zero
    if l_budget == 0:
        l_budget = 1.0
        
    lambda_val = (traj_len - l_budget) / l_budget
    
    if is_success:
        r_l = max(-0.5 * lambda_val + 0.5, 0.1)
    else:
        r_l = min(0.9 * lambda_val - 0.1, -0.1)

    # 3. World Model Interaction Reward
    r_wm = 1.0 - (n_wm / n_wm_max) if n_wm_max > 0 else 1.0
    r_wm = max(0.0, r_wm) # Ensure non-negative

    # Total Reward
    total_reward = alpha * r_success + beta * r_l + gamma * r_wm
    return total_reward

def wm_r1_compute_score(
    is_success: bool,
    traj_len: int,
    n_wm: int,
    n_wm_max: int,
    max_steps: int,
    avg_len_ref: float,
    current_episode: int,
    total_episodes: int,
) -> Dict[str, float]:
    reward = compute_wm_r1_reward(
        is_success, traj_len, n_wm, n_wm_max, max_steps, avg_len_ref, current_episode, total_episodes
    )
    return {
        "overall": reward,
        "is_success": float(is_success),
        "traj_len": float(traj_len),
        "n_wm": float(n_wm),
    }
