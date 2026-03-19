"""
Evaluation metrics for SWOT-GNN: Kling-Gupta Efficiency (KGE).
"""
import numpy as np
from typing import Union


def compute_kge(
    obs: np.ndarray,
    sim: np.ndarray,
) -> float:
    """
    Kling-Gupta Efficiency (KGE).
    KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
    where r=correlation, alpha=std_sim/std_obs, beta=mean_sim/mean_obs.

    Args:
        obs: Observed values
        sim: Simulated/predicted values

    Returns:
        KGE in (-inf, 1], higher is better
    """
    mask = np.isfinite(obs) & np.isfinite(sim)
    obs = np.asarray(obs)[mask]
    sim = np.asarray(sim)[mask]
    if len(obs) < 2:
        return np.nan
    r = np.corrcoef(obs, sim)[0, 1]
    if np.isnan(r):
        return np.nan
    std_obs = np.std(obs)
    std_sim = np.std(sim)
    mean_obs = np.mean(obs)
    mean_sim = np.mean(sim)
    if std_obs < 1e-10:
        return np.nan
    alpha = std_sim / std_obs
    beta = mean_sim / mean_obs if abs(mean_obs) > 1e-10 else 1.0
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return float(kge)
