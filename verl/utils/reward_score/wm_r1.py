
import re
from typing import Dict, List, Tuple
import math
import torch
from collections import defaultdict


def compute_group_stats(
    n_wm_list: List[int],
    traj_len_list: List[int],
    eval_results_list: List[float],
    uid_list,
) -> Dict:
    """按 task_id 分组，计算每组的统计量。

    返回 dict，key=uid，value 包含:
      - c: 成功轨迹数
      - N: 总轨迹数
      - p: c/N
      - L_ref: 成功轨迹平均长度 (全失败时 fallback 到 max(traj_lens))
      - N_WM_ref: 成功轨迹平均 WM 调用次数 (全失败时 fallback 到 max(n_wm_list))
    """
    groups = defaultdict(lambda: {"n_wm": [], "traj_len": [], "success": []})
    for n_wm, traj_len, eval_r, uid in zip(n_wm_list, traj_len_list, eval_results_list, uid_list):
        g = groups[uid]
        g["n_wm"].append(n_wm)
        g["traj_len"].append(traj_len)
        g["success"].append(eval_r > 0.5)

    stats = {}
    for uid, g in groups.items():
        c = sum(g["success"])
        N = len(g["success"])
        p = c / N if N > 0 else 0.0

        success_lens = [l for l, s in zip(g["traj_len"], g["success"]) if s]
        success_nwms = [n for n, s in zip(g["n_wm"], g["success"]) if s]

        L_ref = (sum(success_lens) / len(success_lens)) if success_lens else max(g["traj_len"])
        N_WM_ref = (sum(success_nwms) / len(success_nwms)) if success_nwms else max(g["n_wm"])

        stats[uid] = {
            "c": c,
            "N": N,
            "p": p,
            "L_ref": L_ref,
            "N_WM_ref": N_WM_ref,
        }
    return stats


def compute_wm_r1_reward(
    is_success: bool,
    traj_len: int,
    max_steps: int,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.0,
    p: float = 0.0,
    L_ref: float = 5.0,
    n_wm: int = 0,
    n_wm_max: int = 5,
    N_WM_ref: float = 2.0,
) -> float:
    """
    WM-R1 复合奖励函数。

    R = α · R_success + β · R_L + γ · R_WM

    R_L 和 R_WM 同构: 都用 p = c/N 退火，budget = p · ref + (1-p) · max。
    """
    # ─── 1. Success Reward ───
    r_success = 1.0 if is_success else 0.0

    # ─── 2. Length Reward (DAST) ───
    #   L_budget = p · L_ref + (1-p) · L_max
    l_budget = p * L_ref + (1 - p) * max_steps
    if l_budget == 0:
        l_budget = 1.0
    lambda_val = (traj_len - l_budget) / l_budget

    if is_success:
        r_l = max(-0.5 * lambda_val + 0.5, 0.1)
    else:
        r_l = min(0.9 * lambda_val - 0.1, -0.1)

    # ─── 3. WM Call Count Reward (与 R_L 同构) ───
    #   N_budget = p · N_WM_ref + (1-p) · N_WM_max
    r_wm = 0.0
    if gamma > 0:
        nwm_budget = p * N_WM_ref + (1 - p) * n_wm_max
        if nwm_budget == 0:
            nwm_budget = 1.0
        mu = (n_wm - nwm_budget) / nwm_budget

        if is_success:
            r_wm = max(-0.5 * mu + 0.5, 0.1)
        else:
            r_wm = min(0.9 * mu - 0.1, -0.1)

    # ─── 总奖励 ───
    return alpha * r_success + beta * r_l + gamma * r_wm


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
