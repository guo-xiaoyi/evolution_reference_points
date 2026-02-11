# -*- coding: utf-8 -*-
"""
Created on Fri May 23 11:00:08 2025

@author: XGuo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def value(z, alpha, lambda_, R):
    diff = z - R
    return np.where(diff >= 0, diff ** alpha, -lambda_ * (-diff) ** alpha)
    
def weighting(p, gamma):
    numerator = p ** gamma
    denominator = (p ** gamma + (1 - p) ** gamma) ** (1 / gamma)
    return numerator / denominator


def goldstein_einhorn_weighting(p, gamma, delta):
    numerator = delta * p ** gamma
    denominator = delta * p ** gamma + (1-p) ** gamma
    return numerator / denominator




# Certainty equivalent function
def ce(alpha, lambda_, gamma, r, Z, p, R):


    v = value(Z, alpha, lambda_, R)
    w = weighting(p, gamma)
    y = np.sum(v * w)
    

    if y > 0:
        return y, R + y ** (1 / alpha)
        
    else:
        return y, R - (-y / lambda_) ** (1 / alpha)
        
    
    

def generate_paths(node, path=None, prob=1.0):
    if path is None:
        path = []
    
    current_step = (node["time"], node["value"])
    new_path = path + [current_step]
    new_prob = prob * node["probability"]
    
    if "next" not in node or not node["next"]:
        return [{"path": new_path, "probability": new_prob}]
    
    results = []
    for child in node["next"]:
        results.extend(generate_paths(child, new_path, new_prob))
    
    return results

def value_function(Z, R, alpha, lambda_):
    diff = Z - R
    return np.where(diff >= 0, diff ** alpha, -lambda_ * (-diff) ** alpha)

def weight_function(p, gamma):
    numerator = p ** gamma
    denominator = (p ** gamma + (1 - p) ** gamma) ** (1 / gamma)
    return numerator / denominator

def cumulative_weights(Z, p, R, gamma):
    # Split into gains and losses based on Z - R
    gains = Z >= R
    losses = Z < R
    
    # Sort gains descending, losses ascending
    Z_gains = Z[gains]
    p_gains = p[gains]
    idx_gains = np.argsort(-Z_gains)
    Z_gains = Z_gains[idx_gains]
    p_gains = p_gains[idx_gains]
    
    Z_losses = Z[losses]
    p_losses = p[losses]
    idx_losses = np.argsort(Z_losses)
    Z_losses = Z_losses[idx_losses]
    p_losses = p_losses[idx_losses]
    
    # Compute cumulative weights
    def w(p):
        return (p ** gamma) / ((p ** gamma + (1 - p) ** gamma) ** (1 / gamma))
    
    # Gains
    cum_p_gains = np.cumsum(p_gains)
    cum_p_gains = np.insert(cum_p_gains, 0, 0.0)
    pi_gains = np.array([w(cum_p_gains[i+1]) - w(cum_p_gains[i]) for i in range(len(p_gains))])
    
    # Losses
    cum_p_losses = np.cumsum(p_losses)
    cum_p_losses = np.insert(cum_p_losses, 0, 0.0)
    pi_losses = np.array([w(cum_p_losses[i+1]) - w(cum_p_losses[i]) for i in range(len(p_losses))])
    
    # Reassemble weights in original Z order
    pi = np.zeros_like(Z, dtype=float)
    gain_indices = np.where(gains)[0][idx_gains]
    loss_indices = np.where(losses)[0][idx_losses]
    pi[gain_indices] = pi_gains
    pi[loss_indices] = pi_losses
    
    return pi


def range_ce(alpha, lambda_, gamma, r, Z, p):
    l_min = np.min(Z) * 1.5
    l_max = np.max(Z) * 1.5
    R_range = np.linspace(l_min, l_max, 5000)
    y_ce_pairs = [ce(alpha, lambda_, gamma, r, Z, p, R) for R in R_range]
    y_values, CE_values = zip(*y_ce_pairs)
    y_values = np.array(y_values)
    CE_values = np.array(CE_values)

    ce_df = pd.DataFrame({'R': R_range, 'y': y_values, 'CE': CE_values})

    # Get index mask where y < 0 and R < 20
    mask = (ce_df['y'] < 0) & (ce_df['R'] < 20)
    return R_range, CE_values, mask, y_values, ce_df


def plot_ce_comparison(
    param_name: str,
    param_start: float,
    param_end: float,
    num_points: int,
    fixed_params: dict,
    Z: np.ndarray,
    p: np.ndarray,
    r: float = 1.0
):
    """
    Plot CE curves by varying one of alpha, lambda_, or gamma.

    Parameters
    ----------
    param_name : str
        One of 'alpha', 'lambda_', or 'gamma'.
    param_start : float
        Start value of the varying parameter.
    param_end : float
        End value of the varying parameter.
    num_points : int
        Number of steps to iterate.
    fixed_params : dict
        Dictionary of fixed parameters. Must contain the other two of ['alpha', 'lambda_', 'gamma'].
    Z : np.ndarray
        Outcome array.
    p : np.ndarray
        Probability array.
    r : float
        Discount rate or time preference, default is 1.0.
    """

    # Sanity check
    if param_name not in ['alpha', 'lambda_', 'gamma']:
        raise ValueError("param_name must be one of: 'alpha', 'lambda_', or 'gamma'.")

    # Generate parameter values
    param_values = np.linspace(param_start, param_end, num_points)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    for val in param_values:
        # Dynamically assign parameter values
        alpha = val if param_name == 'alpha' else fixed_params['alpha']
        lambda_ = val if param_name == 'lambda_' else fixed_params['lambda_']
        gamma = val if param_name == 'gamma' else fixed_params['gamma']

        # Get CE curve and value function
        R_range, CE_values, highlight_mask, y_values, ce_df = range_ce(alpha, lambda_, gamma, r, Z, p)

        # Plot CE curve and highlighted range
        ax.plot(R_range, CE_values, color='gray', linewidth=1, alpha=0.7)
        ax.plot(R_range[highlight_mask], CE_values[highlight_mask], color='red', linewidth=1.5)

    # Plot the value function (same for all)
    ax.plot(R_range, y_values, color='black', linewidth=1.5, label='Value Function')

    # Mark the location where y ≈ 0
    closest_to_zero = ce_df.iloc[(ce_df['y'].abs()).argmin()]
    ax.axvline(x=closest_to_zero['R'], color='black', linestyle='--', label='Kink of Value Function')

    # Final plot formatting
    ax.set_xlabel("Reference Point R")
    ax.set_ylabel("Certainty Equivalent")
    # plt.title(f"CE Curves: Varying {param_name} from {param_start} to {param_end}")
    ax.grid(True)
    ax.set_ylim([-50, 50])
    fig.tight_layout()
    ax.legend()
    # Save before showing to avoid backends clearing the figure on show()
    fig.savefig("range_simulation.pdf", format="pdf")
    plt.show()


def partial_adaptation(paths, delta):
    path = paths['path']  # list of tuples: (time, value)
    r_list = []

    for t in range(len(path)):
        numerator = 0.0
        denominator = 0.0
        for s in range(t + 1):
            weight = delta ** (t - s)
            numerator += weight * path[s][1]  # path[s][1] is the value z_s
            denominator += weight
        r_t = numerator / denominator if denominator != 0 else 0.0
        r_list.append(r_t)

    return r_list

def lagged_expectation(node, delta):
    
    path = []
    prob = 1.0
    r_list = []
    current_step = (node["time"], node["value"])
    new_path = path + [current_step]
    new_prob = prob * node["probability"]
    return r_list
