
import re
from typing import Dict, List
import math


def compute_wm_r1_reward(
    is_success: bool,
    traj_len: int,
    max_steps: int,
    avg_len_ref: float,
    current_episode: int,
    total_episodes: int,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.0,
    n_wm: int = 0,
    n_wm_max: int = 5,
    avg_nwm_ref: float = 2.0,
) -> float:
    """
    WM-R1 复合奖励函数。

    R = α · R_success + β · R_L + γ · R_WM

    R_L 和 R_WM 都采用 DAST 动态退火: 训练早期预算宽松，后期收紧。
    """
    # ─── 训练进度 ───
    p = current_episode / total_episodes

    # ─── 1. Success Reward ───
    r_success = 1.0 if is_success else 0.0

    # ─── 2. Length Reward (DAST) ───
    l_budget = p * avg_len_ref + (1 - p) * max_steps
    if l_budget == 0:
        l_budget = 1.0
    lambda_val = (traj_len - l_budget) / l_budget

    if is_success:
        r_l = max(-0.5 * lambda_val + 0.5, 0.1)
    else:
        r_l = min(0.9 * lambda_val - 0.1, -0.1)

    # ─── 3. WM Call Count Reward (同样的退火机制) ───
    #   N_WM_budget = p · avg_nwm_ref + (1-p) · n_wm_max
    #   早期: 允许大量 WM 调用 (budget ≈ n_wm_max)
    #   后期: 鼓励精简 (budget ≈ avg_nwm_ref)
    r_wm = 0.0
    if gamma > 0:
        nwm_budget = p * avg_nwm_ref + (1 - p) * n_wm_max
        if nwm_budget == 0:
            nwm_budget = 1.0
        mu = (n_wm - nwm_budget) / nwm_budget

        if is_success:
            r_wm = max(-0.5 * mu + 0.5, 0.1)
        else:
            r_wm = min(0.9 * mu - 0.1, -0.1)

    # ─── 总奖励 ───
    total_reward = alpha * r_success + beta * r_l + gamma * r_wm
    return total_reward


def wm_r1_compute_score(
    is_success: bool,
    traj_len: int,
    max_steps: int,
    avg_len_ref: float,
    current_episode: int,
    total_episodes: int,
) -> Dict[str, float]:
    reward = 1.0 if is_success else 0.0
    return {
        "overall": reward,
        "is_success": float(is_success),
        "traj_len": float(traj_len),
    }
