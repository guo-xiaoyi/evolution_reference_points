# -*- coding: utf-8 -*-
"""
Discrete Lottery Optimizer - Distinct Reference Points variant

Constraints:
  1. All sub-lottery reference points R (where Y(R)=0) must be pairwise distinct.
  2. Outcome sign must match the requested domain:
       gain  -> all final outcomes z_i > 0
       loss  -> all final outcomes z_i < 0
       mixed -> at least one positive and one negative outcome
  (No EV = 0 constraint.)

The ``--run_all`` flag sweeps all 12 combinations
  (stake: lo/hi) x (outcomes: 4/6) x (sign: gain/loss/mixed)
and collects 5 solutions per combination.

Usage examples (CLI):
  python Simulation/discrete_searcher_distinct_rp.py --outcomes 4 --stake lo --sign mixed
  python Simulation/discrete_searcher_distinct_rp.py --outcomes 6 --stake hi --sign gain
  python Simulation/discrete_searcher_distinct_rp.py --run_all
  python Simulation/discrete_searcher_distinct_rp.py --run_all --cores 4
"""

import numpy as np
from scipy.optimize import fsolve
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import os
import time
import warnings
import multiprocessing as mp
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Optional, Dict, Tuple
import random
from dataclasses import dataclass, field, asdict
import pickle
from scipy.spatial.distance import pdist
import argparse

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False


DEFAULT_RANDOM_SEED = 10
DEFAULT_NUMPY_SEED = 15

ALL_COMBINATIONS: List[Tuple[int, str, str]] = [
    (4, 'lo', 'gain'), (4, 'lo', 'loss'), (4, 'lo', 'mixed'),
    (4, 'hi', 'gain'), (4, 'hi', 'loss'), (4, 'hi', 'mixed'),
    (6, 'lo', 'gain'), (6, 'lo', 'loss'), (6, 'lo', 'mixed'),
    (6, 'hi', 'gain'), (6, 'hi', 'loss'), (6, 'hi', 'mixed'),
]


def apply_global_seeds(seed: Optional[int] = None) -> Tuple[int, int]:
    seed_value = DEFAULT_RANDOM_SEED if seed is None else int(seed)
    numpy_seed = DEFAULT_NUMPY_SEED if seed is None else seed_value
    random.seed(seed_value)
    np.random.seed(numpy_seed)
    return seed_value, numpy_seed


apply_global_seeds()
warnings.filterwarnings('ignore')

alpha_candidates = np.linspace(0.75, 0.95, 5)
lambda_candidates = np.linspace(1.6, 2.5, 3)
gamma_candidates = np.linspace(0.58, 0.70, 5)
r_candidates = np.linspace(0.96, 0.99, 4)

ALT_PARAMS: Optional[List[str]] = [
    f"{alpha},{lam},{gamma}, {r}"
    for alpha in alpha_candidates
    for lam in lambda_candidates
    for gamma in gamma_candidates
    for r in r_candidates
]

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Please install numba to have full performance")
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# ---------- Numba JIT core functions ----------

@jit(nopython=True, cache=True)
def fast_value_function(Z, R, alpha, lambda_):
    diff = Z - R
    result = np.empty_like(diff)
    for i in range(len(diff)):
        if diff[i] >= 0:
            result[i] = (abs(diff[i]) + 1e-12) ** alpha
        else:
            result[i] = -lambda_ * (abs(diff[i]) + 1e-12) ** alpha
    return result


@jit(nopython=True, cache=True)
def fast_probability_weighting(p, gamma):
    p_clipped = max(1e-10, min(p, 1 - 1e-10))
    numerator = p_clipped ** gamma
    denominator = (p_clipped ** gamma + (1 - p_clipped) ** gamma) ** (1 / gamma)
    return numerator / denominator


@jit(nopython=True, cache=True)
def fast_compute_Y(Z, p, R, alpha, lambda_, gamma):
    diff = Z - R
    values = fast_value_function(Z, R, alpha, lambda_)
    total = 0.0

    gains_count = 0
    for i in range(len(diff)):
        if diff[i] >= 0:
            gains_count += 1

    if gains_count > 0:
        gains_Z = np.empty(gains_count, dtype=np.float64)
        gains_p = np.empty(gains_count, dtype=np.float64)
        gains_v = np.empty(gains_count, dtype=np.float64)
        idx = 0
        for i in range(len(diff)):
            if diff[i] >= 0:
                gains_Z[idx] = Z[i]
                gains_p[idx] = p[i]
                gains_v[idx] = values[i]
                idx += 1
        order = np.argsort(gains_Z)
        cum = 0.0
        prev_w = 0.0
        for k in range(gains_count - 1, -1, -1):
            j = order[k]
            cum += gains_p[j]
            w_cum = fast_probability_weighting(cum, gamma)
            dw = w_cum - prev_w
            total += gains_v[j] * dw
            prev_w = w_cum

    losses_count = 0
    for i in range(len(diff)):
        if diff[i] < 0:
            losses_count += 1

    if losses_count > 0:
        losses_Z = np.empty(losses_count, dtype=np.float64)
        losses_p = np.empty(losses_count, dtype=np.float64)
        losses_v = np.empty(losses_count, dtype=np.float64)
        idx = 0
        for i in range(len(diff)):
            if diff[i] < 0:
                losses_Z[idx] = Z[i]
                losses_p[idx] = p[i]
                losses_v[idx] = values[i]
                idx += 1
        order = np.argsort(losses_Z)
        cum = 0.0
        prev_w = 0.0
        for k in range(losses_count):
            j = order[k]
            cum += losses_p[j]
            w_cum = fast_probability_weighting(cum, gamma)
            dw = w_cum - prev_w
            total += losses_v[j] * dw
            prev_w = w_cum

    return total


# ---------- Config & data classes ----------

@dataclass
class OptimizedConfig:
    alpha: float = 0.88
    lambda_: float = 2.25
    gamma: float = 0.61
    r: float = 0.98
    seed: Optional[int] = None

    # Mode
    outcomes: int = 4       # 4 or 6
    stake: str = "lo"       # "hi" or "lo"
    sign: str = "mixed"     # "gain", "loss", or "mixed"

    # Discrete parameter ranges
    lottery_min_bound: int = -101
    lottery_min: int = -26
    lottery_max: int = 26
    lottery_max_bound: int = 101
    prob_choices: List[float] = None

    # Optimization settings
    num_attempts: int = 1000000
    violation_threshold: float = 1.0
    num_cores: Optional[int] = None
    progress_position: int = 0

    # Minimum separation required between any two reference points
    distinct_threshold: float = 0.5

    # Performance settings
    batch_size: int = 10000
    early_termination_solutions: int = 5
    use_fast_prefilter: bool = True

    # Output settings
    output_dir: str = "lottery_results_distinct_rp"
    save_progress_enabled: bool = True
    alt_params: Optional[List[str]] = None
    alt_param_workers: Optional[int] = None

    def __post_init__(self):
        if self.prob_choices is None:
            self.prob_choices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        if self.num_cores is None:
            self.num_cores = mp.cpu_count() or 1
        os.makedirs(self.output_dir, exist_ok=True)
        if self.alt_params is None and 'ALT_PARAMS' in globals():
            self.alt_params = ALT_PARAMS
        if self.alt_param_workers is None:
            self.alt_param_workers = max(1, min(self.num_cores, 4))
        else:
            self.alt_param_workers = max(1, int(self.alt_param_workers))


class CompleteSolution:
    def __init__(self, outcomes: int, params: np.ndarray, violation: float, timestamp: float = None):
        self.outcomes = outcomes
        self.params = params.copy()
        self.violation = violation
        self.timestamp = timestamp or time.time()

        if outcomes == 4:
            self.lottery_values = [int(x) for x in self.params[:8]]
            self.probabilities = list(self.params[8:])
            (self.b11, self.b12, self.c21, self.c22,
             self.c31, self.c32, self.c33, self.c34) = self.lottery_values
            (self.p1, self.p2, self.p3) = self.probabilities
        else:
            self.lottery_values = [int(x) for x in self.params[:11]]
            self.probabilities = list(self.params[11:])
            (self.b11, self.b12, self.c21, self.c22, self.c23,
             self.c31, self.c32, self.c33, self.c34, self.c35, self.c36) = self.lottery_values
            (self.p1, self.p2, self.p3, self.p4, self.p5) = self.probabilities

    def __str__(self):
        return (f"Solution({self.outcomes}out, viol={self.violation:.6f}, "
                f"vals={self.lottery_values}, probs={self.probabilities})")

    def to_dict(self):
        return {
            'outcomes': self.outcomes,
            'params': self.params.tolist(),
            'violation': self.violation,
            'timestamp': self.timestamp,
            'lottery_values': self.lottery_values,
            'probabilities': self.probabilities,
        }


@dataclass
class LotteryStructure:
    Z: np.ndarray
    p: np.ndarray
    outcomes_base: np.ndarray
    expected_value: float


# ---------- Optimizer ----------

class DistinctRPLotteryOptimizer:
    """
    Lottery optimizer with two constraints:
      1. Each sub-lottery reference point R must be pairwise distinct.
      2. Final outcome signs must match ``config.sign`` (gain/loss/mixed).
    No EV = 0 constraint is applied.
    """

    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.alpha = config.alpha
        self.lambda_ = config.lambda_
        self.gamma = config.gamma
        self.r = config.r
        self.outcomes = config.outcomes
        self.stake = config.stake
        self.sign = config.sign

        self._alt_param_specs, self._alt_param_specs_invalid = \
            self._parse_alt_param_specs(self.config.alt_params)
        self._alt_param_workers = (
            min(self.config.alt_param_workers, len(self._alt_param_specs))
            if self._alt_param_specs else 0
        )

        self.stats = {
            'evaluations': 0,
            'solutions_found': 0,
            'best_violation': float('inf'),
            'prefilter_rejections': 0,
            'start_time': None,
            'end_time': None,
        }

        self._prepare_discrete_space()
        self._psutil_process = psutil.Process(os.getpid()) if PSUTIL_AVAILABLE else None
        self._worker_ps = []
        self._resource_sample_interval = 1.0
        self._last_resource_sample = 0.0
        self._cached_resource_status: Optional[Dict[str, str]] = None
        if self._psutil_process:
            self._psutil_process.cpu_percent(None)

    # ---------- Discrete space ----------

    def _prepare_discrete_space(self):
        sign = self.sign
        stake = self.stake
        cfg = self.config

        if stake == 'hi':
            pos_vals = np.arange(cfg.lottery_max, cfg.lottery_max_bound + 1)
            pos_vals = pos_vals[pos_vals % 5 == 0]
            neg_vals = np.arange(cfg.lottery_min_bound, cfg.lottery_min + 1)
            neg_vals = neg_vals[neg_vals % 5 == 0]
            if sign == 'gain':
                self.lottery_values = pos_vals
            elif sign == 'loss':
                self.lottery_values = neg_vals
            else:  # mixed
                self.lottery_values = np.concatenate([neg_vals, pos_vals])
        else:  # lo
            if sign == 'gain':
                self.lottery_values = np.arange(1, cfg.lottery_max + 1)
            elif sign == 'loss':
                self.lottery_values = np.arange(cfg.lottery_min, 0)
            else:  # mixed
                self.lottery_values = np.arange(cfg.lottery_min, cfg.lottery_max + 1)

        self.prob_choices = np.array(self.config.prob_choices)

    # ---------- Alt-param parsing ----------

    def _parse_alt_param_specs(self, raw_specs):
        if not raw_specs:
            return [], False
        parsed = []
        for spec in raw_specs:
            try:
                parts = [x.strip() for x in spec.split(',')]
                if len(parts) not in (3, 4):
                    raise ValueError(f"Invalid alt param format: '{spec}'")
                alpha = float(parts[0])
                lam = float(parts[1])
                gamma = float(parts[2])
                r_override = float(parts[3]) if len(parts) == 4 else None
                parsed.append((alpha, lam, gamma, r_override))
            except Exception:
                warnings.warn(f"Invalid alternate parameter specification skipped: '{spec}'", RuntimeWarning)
                return [], True
        return parsed, False

    # ---------- Progress / resource display ----------

    def _progress_desc(self):
        mode = "Numba" if NUMBA_AVAILABLE else "Python"
        return f"[{self.outcomes}out {self.stake} {self.sign}] [{mode}]"

    def _resource_postfix(self):
        base = {'Numba': 'On' if NUMBA_AVAILABLE else 'Off', 'Cores': self.config.num_cores}
        if not self._psutil_process:
            base['CoreUse'] = 'n/a'
            base['RAM'] = 'n/a'
            return base
        now = time.time()
        if (self._cached_resource_status is not None
                and now - self._last_resource_sample < self._resource_sample_interval):
            cached = dict(self._cached_resource_status)
            cached.update(base)
            return cached
        status = dict(base)
        total_rss = 0
        try:
            total_rss += self._psutil_process.memory_info().rss
        except psutil.Error:
            pass
        live_workers = []
        for wp in self._worker_ps:
            try:
                if wp.is_running():
                    total_rss += wp.memory_info().rss
                    live_workers.append(wp)
            except psutil.Error:
                continue
        self._worker_ps = live_workers
        mem_gb = total_rss / (1024 ** 3) if total_rss else 0.0
        cpu_pct = self._psutil_process.cpu_percent(None)
        core_usage = (cpu_pct / 100.0) * (psutil.cpu_count() or 1)
        status['CoreUse'] = f"{core_usage:.1f}"
        status['RAM'] = f"{mem_gb:.2f}GB"
        if self._worker_ps:
            status['Workers'] = len(self._worker_ps)
        self._cached_resource_status = dict(status)
        self._last_resource_sample = now
        return status

    def _register_solution(self, solution, container):
        container.append(solution)
        self.stats['solutions_found'] += 1
        if solution.violation < self.stats['best_violation']:
            self.stats['best_violation'] = solution.violation

    # ---------- Lottery structure ----------

    def create_lottery_structure(self, params: np.ndarray, r_override: Optional[float] = None):
        r_value = self.r if r_override is None else r_override
        r_sq = r_value * r_value
        if self.outcomes == 4:
            b11, b12, c21, c22, c31, c32, c33, c34, p1, p2, p3 = params
            z1_base = b11 + c21 + c31
            z2_base = b11 + c21 + c32
            z3_base = b12 + c22 + c33
            z4_base = b12 + c22 + c34
            z1 = b11 + c21 * r_value + c31 * r_sq
            z2 = b11 + c21 * r_value + c32 * r_sq
            z3 = b12 + c22 * r_value + c33 * r_sq
            z4 = b12 + c22 * r_value + c34 * r_sq
            prob1 = p1 * p2
            prob2 = p1 * (1 - p2)
            prob3 = (1 - p1) * p3
            prob4 = (1 - p1) * (1 - p3)
            outcomes = np.array([z1, z2, z3, z4])
            probabilities = np.array([prob1, prob2, prob3, prob4])
            outcomes_base = np.array([z1_base, z2_base, z3_base, z4_base], dtype=np.int64)
        else:
            b11, b12, c21, c22, c23, c31, c32, c33, c34, c35, c36, p1, p2, p3, p4, p5 = params
            z1_base = b11 + c21 + c31
            z2_base = b11 + c21 + c32
            z3_base = b12 + c22 + c33
            z4_base = b12 + c22 + c34
            z5_base = b12 + c23 + c35
            z6_base = b12 + c23 + c36
            z1 = b11 + c21 * r_value + c31 * r_sq
            z2 = b11 + c21 * r_value + c32 * r_sq
            z3 = b12 + c22 * r_value + c33 * r_sq
            z4 = b12 + c22 * r_value + c34 * r_sq
            z5 = b12 + c23 * r_value + c35 * r_sq
            z6 = b12 + c23 * r_value + c36 * r_sq
            prob1 = p1 * p2
            prob2 = p1 * (1 - p2)
            prob3 = (1 - p1) * p3 * p4
            prob4 = (1 - p1) * p3 * (1 - p4)
            prob5 = (1 - p1) * (1 - p3) * p5
            prob6 = (1 - p1) * (1 - p3) * (1 - p5)
            outcomes = np.array([z1, z2, z3, z4, z5, z6])
            probabilities = np.array([prob1, prob2, prob3, prob4, prob5, prob6])
            outcomes_base = np.array(
                [z1_base, z2_base, z3_base, z4_base, z5_base, z6_base], dtype=np.int64
            )
        return outcomes, probabilities, outcomes_base

    # ---------- CPT value / R solver ----------

    def _probability_weighting_params(self, p, gamma):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        num = p ** gamma
        den = (p ** gamma + (1 - p) ** gamma) ** (1 / gamma)
        return num / den

    def _compute_Y_python(self, Z, p, R, alpha, lambda_, gamma):
        diff = Z - R
        values = np.where(diff >= 0,
                          np.power(np.abs(diff) + 1e-12, alpha),
                          -lambda_ * np.power(np.abs(diff) + 1e-12, alpha))
        gain_idx = np.where(diff >= 0)[0]
        loss_idx = np.where(diff < 0)[0]
        total = 0.0
        if gain_idx.size > 0:
            order = np.argsort(-Z[gain_idx])
            gp = p[gain_idx][order]
            gv = values[gain_idx][order]
            cum = 0.0; prev_w = 0.0
            for j in range(gp.size):
                cum += float(gp[j])
                w_cum = self._probability_weighting_params(cum, gamma)
                total += gv[j] * (w_cum - prev_w)
                prev_w = w_cum
        if loss_idx.size > 0:
            order = np.argsort(Z[loss_idx])
            lp = p[loss_idx][order]
            lv = values[loss_idx][order]
            cum = 0.0; prev_w = 0.0
            for j in range(lp.size):
                cum += float(lp[j])
                w_cum = self._probability_weighting_params(cum, gamma)
                total += lv[j] * (w_cum - prev_w)
                prev_w = w_cum
        return float(total)

    def _compute_Y(self, Z, p, R):
        if NUMBA_AVAILABLE:
            return float(fast_compute_Y(
                np.asarray(Z, dtype=np.float64), np.asarray(p, dtype=np.float64),
                float(R), self.alpha, self.lambda_, self.gamma))
        return self._compute_Y_python(Z, p, R, self.alpha, self.lambda_, self.gamma)

    def _compute_Y_params(self, Z, p, R, alpha, lambda_, gamma):
        if NUMBA_AVAILABLE:
            return float(fast_compute_Y(
                np.asarray(Z, dtype=np.float64), np.asarray(p, dtype=np.float64),
                float(R), alpha, lambda_, gamma))
        return self._compute_Y_python(Z, p, R, alpha, lambda_, gamma)

    def find_R_where_Y_equals_zero(self, Z, p):
        def eq(R):
            return self._compute_Y(Z, p, R)
        for start in [0.0, float(np.mean(Z)), float(np.median(Z))]:
            try:
                sol = fsolve(eq, start, xtol=1e-6)[0]
                if abs(eq(sol)) < 1e-4:
                    return float(sol)
            except Exception:
                continue
        return None

    def find_R_where_Y_equals_zero_params(self, Z, p, alpha, lambda_, gamma):
        def eq(R):
            return self._compute_Y_params(Z, p, R, alpha, lambda_, gamma)
        for start in [0.0, float(np.mean(Z)), float(np.median(Z))]:
            try:
                sol = fsolve(eq, start, xtol=1e-6)[0]
                if abs(eq(sol)) < 1e-4:
                    return float(sol)
            except Exception:
                continue
        return None

    # ---------- Sign constraint ----------

    def _sign_violation(self, Z: np.ndarray) -> float:
        """Return penalty when outcome signs don't match self.sign."""
        sign = self.sign
        if sign == 'gain':
            # All outcomes must be strictly positive
            return float(sum(max(0.0, -z + 0.01) for z in Z))
        elif sign == 'loss':
            # All outcomes must be strictly negative
            return float(sum(max(0.0, z + 0.01) for z in Z))
        else:  # mixed
            has_pos = any(z > 0 for z in Z)
            has_neg = any(z < 0 for z in Z)
            if not has_pos or not has_neg:
                return 100.0
            return 0.0

    # ---------- Distinct-RP violation ----------

    def _distinct_rp_violation(self, R_values: List[Optional[float]]) -> float:
        threshold = self.config.distinct_threshold
        total = 0.0
        valid = []
        for R in R_values:
            if R is None:
                total += 50.0
            else:
                valid.append(R)
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                sep = abs(valid[i] - valid[j])
                if sep < threshold:
                    total += (threshold - sep)
        return total

    # ---------- Sub-lottery R collection ----------

    def _collect_R_values(self, Z, p, params) -> List[Optional[float]]:
        R_full = self.find_R_where_Y_equals_zero(Z, p)
        if self.outcomes == 4:
            z1, z2, z3, z4 = Z
            p2, p3 = params[9], params[10]
            Z_L1 = np.array([z1, z2]); p_L1 = np.array([p2, 1 - p2])
            Z_L2 = np.array([z3, z4]); p_L2 = np.array([p3, 1 - p3])
            return [R_full,
                    self.find_R_where_Y_equals_zero(Z_L1, p_L1),
                    self.find_R_where_Y_equals_zero(Z_L2, p_L2)]
        else:
            z1, z2, z3, z4, z5, z6 = Z
            p2, p3, p4, p5 = params[12], params[13], params[14], params[15]
            Z_L1 = np.array([z1, z2]); p_L1 = np.array([p2, 1 - p2])
            Z_L2 = np.array([z3, z4]); p_L2 = np.array([p4, 1 - p4])
            Z_L3 = np.array([z5, z6]); p_L3 = np.array([p5, 1 - p5])
            Z_L4 = np.array([z3, z4, z5, z6])
            p_L4 = np.array([p3 * p4, p3 * (1 - p4), (1 - p3) * p5, (1 - p3) * (1 - p5)])
            return [R_full,
                    self.find_R_where_Y_equals_zero(Z_L1, p_L1),
                    self.find_R_where_Y_equals_zero(Z_L2, p_L2),
                    self.find_R_where_Y_equals_zero(Z_L3, p_L3),
                    self.find_R_where_Y_equals_zero(Z_L4, p_L4)]

    def _collect_R_values_params(self, Z, p, params, alpha, lambda_, gamma):
        R_full = self.find_R_where_Y_equals_zero_params(Z, p, alpha, lambda_, gamma)
        if self.outcomes == 4:
            z1, z2, z3, z4 = Z
            p2, p3 = params[9], params[10]
            Z_L1 = np.array([z1, z2]); p_L1 = np.array([p2, 1 - p2])
            Z_L2 = np.array([z3, z4]); p_L2 = np.array([p3, 1 - p3])
            return [R_full,
                    self.find_R_where_Y_equals_zero_params(Z_L1, p_L1, alpha, lambda_, gamma),
                    self.find_R_where_Y_equals_zero_params(Z_L2, p_L2, alpha, lambda_, gamma)]
        else:
            z1, z2, z3, z4, z5, z6 = Z
            p2, p3, p4, p5 = params[12], params[13], params[14], params[15]
            Z_L1 = np.array([z1, z2]); p_L1 = np.array([p2, 1 - p2])
            Z_L2 = np.array([z3, z4]); p_L2 = np.array([p4, 1 - p4])
            Z_L3 = np.array([z5, z6]); p_L3 = np.array([p5, 1 - p5])
            Z_L4 = np.array([z3, z4, z5, z6])
            p_L4 = np.array([p3 * p4, p3 * (1 - p4), (1 - p3) * p5, (1 - p3) * (1 - p5)])
            return [R_full,
                    self.find_R_where_Y_equals_zero_params(Z_L1, p_L1, alpha, lambda_, gamma),
                    self.find_R_where_Y_equals_zero_params(Z_L2, p_L2, alpha, lambda_, gamma),
                    self.find_R_where_Y_equals_zero_params(Z_L3, p_L3, alpha, lambda_, gamma),
                    self.find_R_where_Y_equals_zero_params(Z_L4, p_L4, alpha, lambda_, gamma)]

    # ---------- Constraints ----------

    def _basic_constraint_violation(self, params, return_structure: bool = False):
        """Fast prefilter: probability bounds + sign check only."""
        if self.outcomes == 4:
            p1, p2, p3 = params[8:]
            if not (0 <= p1 <= 1 and 0 <= p2 <= 1 and 0 <= p3 <= 1):
                return (True, 1000.0, None) if return_structure else (True, 1000.0)
        else:
            p1, p2, p3, p4, p5 = params[11:]
            if not all(0 <= pp <= 1 for pp in [p1, p2, p3, p4, p5]):
                return (True, 1000.0, None) if return_structure else (True, 1000.0)

        Z, p, outcomes_base = self.create_lottery_structure(params)
        expected_value = float(np.sum(Z * p))
        sign_viol = self._sign_violation(Z)
        total_viol = sign_viol

        if return_structure:
            structure = LotteryStructure(Z=Z, p=p, outcomes_base=outcomes_base,
                                         expected_value=expected_value)
            return (sign_viol > 0), total_viol, structure
        return (sign_viol > 0), total_viol

    def check_full_constraints(self, params, track_stats: bool = True,
                               precomputed: Optional[LotteryStructure] = None):
        """
        Full constraint check:
          1. Sign constraint (gain / loss / mixed)
          2. All sub-lottery reference points pairwise distinct
          3. Probability bounds soft penalty
        No EV = 0 constraint.
        """
        try:
            if track_stats:
                self.stats['evaluations'] += 1
            if precomputed is None:
                Z, p, outcomes_base = self.create_lottery_structure(params)
                expected_value = float(np.sum(Z * p))
                structure = LotteryStructure(Z=Z, p=p, outcomes_base=outcomes_base,
                                             expected_value=expected_value)
            else:
                structure = precomputed
                Z = structure.Z
                p = structure.p
                expected_value = structure.expected_value

            violations = {}
            total_violations = 0.0

            # 1. Sign constraint
            sign_viol = self._sign_violation(Z)
            violations['sign'] = sign_viol
            total_violations += sign_viol

            # 2. Distinct reference points
            R_values = self._collect_R_values(Z, p, params)
            distinct_viol = self._distinct_rp_violation(R_values)
            violations['distinct_rp'] = distinct_viol
            total_violations += distinct_viol

            # 3. Probability bounds soft penalty
            prob_viol = 0.0
            for prob in (params[8:] if self.outcomes == 4 else params[11:]):
                if prob < 0:
                    prob_viol += (-prob)
                elif prob > 1:
                    prob_viol += (prob - 1)
            violations['prob_bounds'] = prob_viol
            total_violations += prob_viol

            violations['total'] = float(total_violations)
            if total_violations < self.stats['best_violation']:
                self.stats['best_violation'] = float(total_violations)
            return violations, True, structure
        except Exception as e:
            return {'total': 10000, 'error': str(e)}, False, None

    def check_constraints_with_params(self, params, alpha, lambda_, gamma,
                                      r_override: Optional[float] = None,
                                      base_structure: Optional[LotteryStructure] = None):
        """Alt-parameter version."""
        try:
            if base_structure is not None and r_override is None:
                Z = base_structure.Z
                p = base_structure.p
            else:
                Z, p, _ = self.create_lottery_structure(params, r_override)

            total_violations = self._sign_violation(Z)
            R_values = self._collect_R_values_params(Z, p, params, alpha, lambda_, gamma)
            total_violations += self._distinct_rp_violation(R_values)

            prob_viol = 0.0
            for prob in (params[8:] if self.outcomes == 4 else params[11:]):
                if prob < 0:
                    prob_viol += (-prob)
                elif prob > 1:
                    prob_viol += (prob - 1)
            total_violations += prob_viol
            return {'total': float(total_violations)}, True
        except Exception as e:
            return {'total': 10000, 'error': str(e)}, False

    def _evaluate_alt_param_spec(self, params, spec, base_structure=None):
        alpha, lambda_, gamma, r_override = spec
        hint = base_structure if (base_structure is not None and r_override is None) else None
        alt_viol, alt_ok = self.check_constraints_with_params(
            params, alpha, lambda_, gamma, r_override, base_structure=hint)
        if not alt_ok:
            return False
        return alt_viol.get('total', float('inf')) < self.config.violation_threshold

    def _alt_param_worker_task(self, params, specs, stop_event, base_structure):
        for spec in specs:
            if stop_event.is_set():
                return True
            if not self._evaluate_alt_param_spec(params, spec, base_structure):
                stop_event.set()
                return False
        return True

    def _alt_params_valid(self, params, base_structure=None):
        if self._alt_param_specs_invalid:
            return False
        if not self._alt_param_specs:
            return True
        if self._alt_param_workers <= 1 or len(self._alt_param_specs) == 1:
            for spec in self._alt_param_specs:
                if not self._evaluate_alt_param_spec(params, spec, base_structure):
                    return False
            return True
        worker_count = min(self._alt_param_workers, len(self._alt_param_specs))
        stop_event = threading.Event()
        chunks = [[] for _ in range(worker_count)]
        for idx, spec in enumerate(self._alt_param_specs):
            chunks[idx % worker_count].append(spec)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(self._alt_param_worker_task, params, chunk, stop_event, base_structure)
                for chunk in chunks if chunk
            ]
            for future in as_completed(futures):
                if not future.result():
                    stop_event.set()
                    return False
        return True

    # ---------- Batch generation ----------

    def generate_batch_params(self, batch_size: int) -> np.ndarray:
        if self.outcomes == 4:
            lottery_draws = np.random.choice(
                self.lottery_values, size=(batch_size, 8)).astype(np.float64, copy=False)
            prob_draws = np.random.choice(
                self.prob_choices, size=(batch_size, 3)).astype(np.float64, copy=False)
        else:
            lottery_draws = np.random.choice(
                self.lottery_values, size=(batch_size, 11)).astype(np.float64, copy=False)
            prob_draws = np.random.choice(
                self.prob_choices, size=(batch_size, 5)).astype(np.float64, copy=False)
        return np.concatenate([lottery_draws, prob_draws], axis=1)

    def fast_prefilter(self, params_batch: np.ndarray, cache_structures: bool = False):
        if not self.config.use_fast_prefilter:
            return params_batch, None
        valid_indices = []
        cached_structures = [] if cache_structures else None
        for i in range(len(params_batch)):
            params = params_batch[i]
            if cache_structures:
                invalid, violation, structure = self._basic_constraint_violation(
                    params, return_structure=True)
            else:
                invalid, violation = self._basic_constraint_violation(params)
                structure = None
            if not invalid:
                valid_indices.append(i)
                if cached_structures is not None and structure is not None:
                    cached_structures.append(structure)
            else:
                self.stats['prefilter_rejections'] += 1
        shape = (0, 11) if self.outcomes == 4 else (0, 16)
        filtered = params_batch[valid_indices] if valid_indices else np.empty(shape)
        return filtered, cached_structures

    # ---------- Optimization loop ----------

    def batch_optimize(self) -> List[CompleteSolution]:
        solutions = []
        attempts_made = 0
        self.stats['start_time'] = time.time()
        use_parallel = (self.config.num_cores or 1) > 1
        pool = None
        pool_terminated = False

        if use_parallel:
            config_state = asdict(self.config)
            config_state['num_cores'] = 1
            ctx = mp.get_context('spawn') if os.name == 'nt' else mp.get_context()
            pool = ctx.Pool(
                processes=self.config.num_cores,
                initializer=_mp_worker_initializer,
                initargs=(config_state,)
            )
            if PSUTIL_AVAILABLE:
                handles = []
                for proc in getattr(pool, "_pool", []):
                    try:
                        ps_proc = psutil.Process(proc.pid)
                        ps_proc.cpu_percent(None)
                        handles.append(ps_proc)
                    except (psutil.Error, AttributeError):
                        continue
                self._worker_ps = handles
        else:
            self._worker_ps = []

        pbar = tqdm(
            total=self.config.num_attempts,
            desc=self._progress_desc(),
            position=self.config.progress_position,
            leave=True,
            dynamic_ncols=True,
        )
        early_stop = False
        try:
            while attempts_made < self.config.num_attempts and not early_stop:
                current_batch = min(self.config.batch_size,
                                    self.config.num_attempts - attempts_made)
                params_batch = self.generate_batch_params(current_batch)
                filtered_params, cached_structs = self.fast_prefilter(
                    params_batch, cache_structures=not use_parallel)

                if use_parallel and filtered_params.size > 0:
                    rows = len(filtered_params)
                    block_size = max(1, min(rows, rows // max(1, self.config.num_cores * 2)))
                    chunk_iter = (
                        filtered_params[s:s + block_size]
                        for s in range(0, rows, block_size)
                    )
                    for result in pool.imap_unordered(_mp_worker_process, chunk_iter, chunksize=1):
                        if not result:
                            continue
                        if isinstance(result, dict) and 'error' in result:
                            warnings.warn(f"Worker error: {result['error']}")
                            continue
                        self.stats['evaluations'] += result.get('evaluations', 0)
                        for violation_value, candidate_params in result.get('solutions', []):
                            sol = CompleteSolution(self.outcomes, candidate_params, violation_value)
                            self._register_solution(sol, solutions)
                            if len(solutions) >= self.config.early_termination_solutions:
                                early_stop = True
                                pool_terminated = True
                                pool.terminate()
                                break
                        if early_stop:
                            break
                else:
                    if cached_structs is not None and len(cached_structs) != len(filtered_params):
                        cached_iter = None
                    else:
                        cached_iter = cached_structs
                    param_iter = (
                        zip(filtered_params, cached_iter)
                        if cached_iter is not None
                        else ((p, None) for p in filtered_params)
                    )
                    for params, cached_structure in param_iter:
                        violations, valid, structure = self.check_full_constraints(
                            params, precomputed=cached_structure)
                        if not (valid and violations['total'] < self.config.violation_threshold):
                            continue
                        if not self._alt_params_valid(params, structure):
                            continue
                        sol = CompleteSolution(self.outcomes, params, violations['total'])
                        self._register_solution(sol, solutions)
                        if len(solutions) >= self.config.early_termination_solutions:
                            early_stop = True
                            break

                attempts_made += current_batch
                pbar.update(current_batch)
                pbar.set_postfix({
                    'Solutions': len(solutions),
                    'Best': f"{self.stats['best_violation']:.3f}",
                    'Prefilter': self.stats['prefilter_rejections'],
                    **self._resource_postfix(),
                })
        finally:
            pbar.close()
            self.stats['end_time'] = time.time()
            if pool:
                if pool_terminated:
                    pool.join()
                else:
                    pool.close()
                    pool.join()
            self._worker_ps = []
        return solutions

    def solve(self) -> List[CompleteSolution]:
        print(f"\n{'='*60}")
        print(f"Distinct-RP Optimizer | outcomes={self.outcomes} stake={self.stake} sign={self.sign}")
        print(f"  distinct_threshold={self.config.distinct_threshold}")
        print(f"  Numba: {'Enabled' if NUMBA_AVAILABLE else 'Disabled'}")
        print(f"  CPT params: alpha={self.alpha} lambda={self.lambda_} gamma={self.gamma} r={self.r}")
        solutions = self.batch_optimize()
        solutions.sort(key=lambda s: s.violation)
        elapsed = (self.stats['end_time'] or time.time()) - self.stats['start_time']
        print(f"  Done in {elapsed:.2f}s | evaluations={self.stats['evaluations']:,} | "
              f"solutions={len(solutions)}")
        return solutions

    # ---------- Reporting ----------

    def calculate_solution_diversity(self, solutions):
        if len(solutions) < 2:
            return {'diversity': 0.0, 'num_solutions': len(solutions)}
        params_array = np.array([s.params for s in solutions])
        distances = pdist(params_array, metric='euclidean')
        return {
            'num_solutions': len(solutions),
            'mean_distance': float(np.mean(distances)),
            'min_distance': float(np.min(distances)),
            'max_distance': float(np.max(distances)),
            'std_distance': float(np.std(distances)),
            'unique_lottery_combinations': len(set(tuple(s.lottery_values) for s in solutions)),
            'unique_prob_combinations': len(set(tuple(s.probabilities) for s in solutions)),
        }

    def _default_filename(self) -> str:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        seed_sfx = f"_seed{self.config.seed}" if self.config.seed is not None else ""
        tag = f"{self.outcomes}out_{self.stake}_{self.sign}"
        return os.path.join(self.config.output_dir, f"distinct_rp_{tag}{seed_sfx}_{ts}.txt")

    def save_solutions_to_file(self, solutions, filename: str = None) -> str:
        if filename is None:
            filename = self._default_filename()
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write(f"Distinct-RP Lottery Optimizer | {self.outcomes} outcomes | "
                        f"stake={self.stake} | sign={self.sign}\n")
                f.write("=" * 80 + "\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Constraints: distinct reference points (threshold={self.config.distinct_threshold}), "
                        f"sign={self.sign}. No EV=0 constraint.\n")
                f.write(f"Solutions found: {len(solutions)}\n")
                f.write(f"CPT params: alpha={self.alpha}, lambda={self.lambda_}, "
                        f"gamma={self.gamma}, r={self.r}\n")
                f.write(f"Lottery range: ")
                if self.stake == 'hi':
                    if self.sign == 'gain':
                        f.write(f"[{self.config.lottery_max}, {self.config.lottery_max_bound}] (×5)\n")
                    elif self.sign == 'loss':
                        f.write(f"[{self.config.lottery_min_bound}, {self.config.lottery_min}] (×5)\n")
                    else:
                        f.write(f"[{self.config.lottery_min_bound}, {self.config.lottery_min}] ∪ "
                                f"[{self.config.lottery_max}, {self.config.lottery_max_bound}] (×5)\n")
                else:
                    if self.sign == 'gain':
                        f.write(f"[1, {self.config.lottery_max}]\n")
                    elif self.sign == 'loss':
                        f.write(f"[{self.config.lottery_min}, -1]\n")
                    else:
                        f.write(f"[{self.config.lottery_min}, {self.config.lottery_max}]\n")
                f.write(f"Probability choices: {self.config.prob_choices}\n")
                f.write(f"Violation threshold: {self.config.violation_threshold}\n")
                f.write(f"Total evaluations: {self.stats['evaluations']:,}\n")
                f.write("\n")

                if not solutions:
                    f.write("No solutions found!\n")
                    f.write("Suggestions: increase --attempts, relax --violation_threshold "
                            "or --distinct_threshold\n")
                    return filename

                for i, solution in enumerate(solutions):
                    Z, p, Z_base = self.create_lottery_structure(solution.params)
                    expected_value = float(np.sum(Z * p))
                    f.write(f"\n{'='*60}\nSolution {i+1}\n{'='*60}\n")
                    f.write(f"Violation: {solution.violation:.6f}\n")
                    f.write(f"Timestamp: {datetime.fromtimestamp(solution.timestamp).strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                    if self.outcomes == 4:
                        f.write(f"  Stage 1: b11={solution.b11:4d}, b12={solution.b12:4d}\n")
                        f.write(f"  Stage 2: c21={solution.c21:4d}, c22={solution.c22:4d}\n")
                        f.write(f"  Stage 3: c31={solution.c31:4d}, c32={solution.c32:4d}, "
                                f"c33={solution.c33:4d}, c34={solution.c34:4d}\n")
                        f.write(f"  Probs:   p1={solution.p1:.3f}, p2={solution.p2:.3f}, "
                                f"p3={solution.p3:.3f}\n")
                    else:
                        f.write(f"  Stage 1: b11={solution.b11:4d}, b12={solution.b12:4d}\n")
                        f.write(f"  Stage 2: c21={solution.c21:4d}, c22={solution.c22:4d}, "
                                f"c23={solution.c23:4d}\n")
                        f.write(f"  Stage 3: c31={solution.c31:4d}, c32={solution.c32:4d}, "
                                f"c33={solution.c33:4d}, c34={solution.c34:4d}, "
                                f"c35={solution.c35:4d}, c36={solution.c36:4d}\n")
                        f.write(f"  Probs:   p1={solution.p1:.3f}, p2={solution.p2:.3f}, "
                                f"p3={solution.p3:.3f}, p4={solution.p4:.3f}, "
                                f"p5={solution.p5:.3f}\n")

                    f.write("\nFinal outcomes (nominal -> discounted):\n")
                    for idx, (z_nom, z_disc, prob) in enumerate(zip(Z_base, Z, p), 1):
                        f.write(f"  z{idx}: {int(z_nom):6d} -> {z_disc:8.3f}  (prob={prob:.3f})\n")
                    f.write(f"  E[Z] = {expected_value:.6f}\n")

                    # Reference points
                    f.write("\nReference points:\n")
                    R_values = self._collect_R_values(Z, p, solution.params)
                    labels = (["R_full", "R_L1", "R_L2"]
                              if self.outcomes == 4
                              else ["R_full", "R_L1", "R_L2", "R_L3", "R_L4"])
                    for lbl, R in zip(labels, R_values):
                        f.write(f"  {lbl:8s} = {R:.4f}\n" if R is not None
                                else f"  {lbl:8s} = not found\n")
                    valid_R = [v for v in R_values if v is not None]
                    all_distinct = all(
                        abs(valid_R[ii] - valid_R[jj]) >= self.config.distinct_threshold
                        for ii in range(len(valid_R))
                        for jj in range(ii + 1, len(valid_R))
                    )
                    f.write(f"  Pairwise distinct (≥{self.config.distinct_threshold}): "
                            f"{'✓' if all_distinct else '✗'}\n")

                    # Sign check
                    sign_ok = self._sign_violation(Z) == 0.0
                    f.write(f"  Sign constraint ({self.sign}): {'✓' if sign_ok else '✗'}\n")

                # Summary table
                f.write(f"\n{'='*80}\nSolution Summary Table\n{'='*80}\n")
                rows = []
                for i, sol in enumerate(solutions):
                    Z, p, Z_base = self.create_lottery_structure(sol.params)
                    row = {'Sol': i + 1}
                    if self.outcomes == 4:
                        row.update(dict(zip(
                            ['b11','b12','c21','c22','c31','c32','c33','c34'],
                            sol.lottery_values)))
                        row.update({'p1': sol.p1, 'p2': sol.p2, 'p3': sol.p3})
                        for k, v in enumerate(Z_base, 1):
                            row[f'z{k}'] = int(v)
                        for k, v in enumerate(Z, 1):
                            row[f'z{k}_d'] = float(v)
                    else:
                        row.update(dict(zip(
                            ['b11','b12','c21','c22','c23','c31','c32','c33','c34','c35','c36'],
                            sol.lottery_values)))
                        row.update({'p1': sol.p1, 'p2': sol.p2, 'p3': sol.p3,
                                    'p4': sol.p4, 'p5': sol.p5})
                        for k, v in enumerate(Z_base, 1):
                            row[f'z{k}'] = int(v)
                        for k, v in enumerate(Z, 1):
                            row[f'z{k}_d'] = float(v)
                    row['E[Z]'] = float(np.sum(Z * p))
                    row['Violation'] = sol.violation
                    rows.append(row)
                f.write(pd.DataFrame(rows).to_string(index=False, float_format='%.3f'))
                f.write("\n")

                # Diversity
                div = self.calculate_solution_diversity(solutions)
                f.write(f"\n{'='*80}\nDiversity\n{'='*80}\n")
                f.write(f"Solutions: {div['num_solutions']}\n")
                if len(solutions) >= 2:
                    f.write(f"Unique lottery combos: {div['unique_lottery_combinations']}\n")
                    f.write(f"Unique prob combos:    {div['unique_prob_combinations']}\n")
                    f.write(f"Mean pairwise dist:    {div.get('mean_distance',0):.3f}\n")
                    f.write(f"Dist range:            [{div.get('min_distance',0):.3f}, "
                            f"{div.get('max_distance',0):.3f}]\n")

                # Performance
                elapsed = (self.stats['end_time'] or time.time()) - self.stats['start_time']
                f.write(f"\n{'='*80}\nPerformance\n{'='*80}\n")
                f.write(f"Runtime: {elapsed:.2f}s\n")
                f.write(f"Evaluations: {self.stats['evaluations']:,}\n")
                f.write(f"Solutions found: {self.stats['solutions_found']}\n")
                if elapsed > 0:
                    f.write(f"Speed: {self.stats['evaluations'] / elapsed:.0f} evals/s\n")
                if solutions:
                    v_list = [s.violation for s in solutions]
                    f.write(f"Best violation:  {min(v_list):.6f}\n")
                    f.write(f"Mean violation:  {np.mean(v_list):.6f}\n")
                    f.write(f"Worst violation: {max(v_list):.6f}\n")
            return filename
        except Exception as e:
            print(f"Error saving file: {e}")
            return None

    def save_progress(self, solutions, filename: str = None):
        if not self.config.save_progress_enabled:
            return None
        if filename is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            tag = f"{self.outcomes}out_{self.stake}_{self.sign}"
            seed_sfx = f"_seed{self.config.seed}" if self.config.seed is not None else ""
            filename = os.path.join(
                self.config.output_dir,
                f"distinct_rp_progress_{tag}{seed_sfx}_{ts}.pkl"
            )
        try:
            with open(filename, 'wb') as f:
                pickle.dump({
                    'solutions': [s.to_dict() for s in solutions],
                    'stats': self.stats,
                    'config': self.config,
                    'timestamp': datetime.now(),
                }, f)
            print(f"Progress saved: {filename}")
            return filename
        except Exception as e:
            print(f"Error saving progress: {e}")
            return None


# ---------- Multiprocessing helpers ----------

MP_WORKER_OPTIMIZER = None


def _mp_worker_initializer(config_state: Dict):
    state_copy = dict(config_state)
    state_copy['num_cores'] = 1
    config = OptimizedConfig(**state_copy)
    global MP_WORKER_OPTIMIZER
    MP_WORKER_OPTIMIZER = DistinctRPLotteryOptimizer(config)


def _mp_worker_process(params_block):
    optimizer = MP_WORKER_OPTIMIZER
    if optimizer is None:
        return {'error': 'Worker not initialized'}
    evaluations = 0
    solutions = []
    try:
        for params in params_block:
            evaluations += 1
            violations, valid, structure = optimizer.check_full_constraints(
                params, track_stats=False)
            if not (valid and violations['total'] < optimizer.config.violation_threshold):
                continue
            if not optimizer._alt_params_valid(params, structure):
                continue
            solutions.append((float(violations['total']), params.copy()))
        return {'evaluations': evaluations, 'solutions': solutions}
    except Exception as exc:
        return {'error': str(exc)}


# ---------- Preset configs ----------

def build_preset_config(outcomes: int, stake: str, sign: str = 'mixed') -> OptimizedConfig:
    base = dict(
        outcomes=outcomes,
        stake=stake,
        sign=sign,
        num_attempts=10_000_000,
        violation_threshold=1.0,
        early_termination_solutions=5,
        output_dir="lottery_results_distinct_rp",
    )
    if outcomes == 4 and stake == 'hi':
        base.update(lottery_min=-100, lottery_max=100,
                    lottery_min_bound=-1000, lottery_max_bound=1000,
                    batch_size=10000)
    elif outcomes == 4 and stake == 'lo':
        base.update(lottery_min=-50, lottery_max=50,
                    batch_size=10000)
    elif outcomes == 6 and stake == 'hi':
        base.update(lottery_min=-500, lottery_max=500,
                    lottery_min_bound=-1000, lottery_max_bound=1000,
                    batch_size=30000,
                    output_dir="lottery_results_distinct_rp_6outcomes")
    else:  # 6 lo
        base.update(lottery_min=-50, lottery_max=50,
                    batch_size=10000,
                    output_dir="lottery_results_distinct_rp_6outcomes")
    return OptimizedConfig(**base)


# ---------- Seed-parallel job ----------

def run_seed_job(config_state: Dict, seed_value: Optional[int], position: int = 0):
    applied_seed, numpy_seed = apply_global_seeds(seed_value)
    state_copy = dict(config_state)
    state_copy['seed'] = applied_seed
    state_copy['progress_position'] = position
    config = OptimizedConfig(**state_copy)
    optimizer = DistinctRPLotteryOptimizer(config)
    solutions = optimizer.solve()
    txt_file = optimizer.save_solutions_to_file(solutions)
    progress_file = optimizer.save_progress(solutions) if optimizer.config.save_progress_enabled else None
    return {
        'seed': applied_seed,
        'numpy_seed': numpy_seed,
        'solutions_found': len(solutions),
        'best_violation': solutions[0].violation if solutions else None,
        'txt_file': txt_file,
        'progress_file': progress_file,
    }


# ---------- Run all combinations ----------

def run_all_combinations(cores_per_run: int = None,
                         num_attempts: int = None,
                         violation_threshold: float = None,
                         distinct_threshold: float = None,
                         solutions_per_combo: int = 5,
                         seed: Optional[int] = None) -> Dict:
    """
    Run all 12 combinations (lo/hi) x (4/6) x (gain/loss/mixed),
    collecting ``solutions_per_combo`` solutions each.

    Returns a dict keyed by (outcomes, stake, sign).
    """
    summary_rows = []
    all_results = {}

    print(f"\n{'='*70}")
    print(f"RUNNING ALL {len(ALL_COMBINATIONS)} COMBINATIONS")
    print(f"  solutions per combination : {solutions_per_combo}")
    print(f"  cores per run             : {cores_per_run or 'auto'}")
    print(f"{'='*70}\n")

    for combo_idx, (outcomes, stake, sign) in enumerate(ALL_COMBINATIONS, 1):
        print(f"\n[{combo_idx}/{len(ALL_COMBINATIONS)}] outcomes={outcomes} stake={stake} sign={sign}")
        config = build_preset_config(outcomes, stake, sign)
        config.early_termination_solutions = solutions_per_combo
        if cores_per_run is not None:
            config.num_cores = max(1, cores_per_run)
        if num_attempts is not None:
            config.num_attempts = num_attempts
        if violation_threshold is not None:
            config.violation_threshold = violation_threshold
        if distinct_threshold is not None:
            config.distinct_threshold = distinct_threshold
        if seed is not None:
            applied_seed, _ = apply_global_seeds(seed)
            config.seed = applied_seed

        optimizer = DistinctRPLotteryOptimizer(config)
        solutions = optimizer.solve()
        txt_file = optimizer.save_solutions_to_file(solutions)
        if config.save_progress_enabled:
            optimizer.save_progress(solutions)

        key = (outcomes, stake, sign)
        all_results[key] = {'solutions': solutions, 'file': txt_file, 'optimizer': optimizer}

        summary_rows.append({
            'outcomes': outcomes, 'stake': stake, 'sign': sign,
            'found': len(solutions),
            'best_violation': f"{solutions[0].violation:.4f}" if solutions else 'N/A',
            'file': os.path.basename(txt_file) if txt_file else 'N/A',
        })

    # Print summary
    print(f"\n{'='*70}")
    print("ALL-COMBINATIONS SUMMARY")
    print(f"{'='*70}")
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"{'='*70}\n")

    return all_results


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="Discrete Lottery Optimizer - Distinct Reference Points")
    parser.add_argument("--outcomes", type=int, choices=[4, 6], default=4)
    parser.add_argument("--stake", type=str, choices=["hi", "lo"], default="lo")
    parser.add_argument("--sign", type=str, choices=["gain", "loss", "mixed"], default="mixed",
                        help="Outcome sign domain: gain (all z>0), loss (all z<0), mixed")
    parser.add_argument("--run_all", action="store_true",
                        help="Run all 12 combinations (lo/hi)x(4/6)x(gain/loss/mixed)")
    parser.add_argument("--attempts", type=int, default=None)
    parser.add_argument("--violation_threshold", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--early", type=int, default=None,
                        help="Early termination: stop after this many solutions (default 5)")
    parser.add_argument("--distinct_threshold", type=float, default=None,
                        help="Min separation between reference points (default 0.5)")
    parser.add_argument("--min", dest="lot_min", type=int, default=None)
    parser.add_argument("--max", dest="lot_max", type=int, default=None)
    parser.add_argument("--min_bound", type=int, default=None)
    parser.add_argument("--max_bound", type=int, default=None)
    parser.add_argument("--save_progress", action="store_true")
    parser.add_argument("--alt_params", nargs='*', default=None)
    parser.add_argument("--alt_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, nargs='*', default=None)
    parser.add_argument("--seed_workers", type=int, default=None)
    parser.add_argument("--cores", type=int, default=None)
    args = parser.parse_args()

    # ---- run_all mode ----
    if args.run_all:
        run_all_combinations(
            cores_per_run=args.cores,
            num_attempts=args.attempts,
            violation_threshold=args.violation_threshold,
            distinct_threshold=args.distinct_threshold,
            solutions_per_combo=args.early or 5,
            seed=args.seed[0] if args.seed else None,
        )
        return

    # ---- single combination mode ----
    config = build_preset_config(args.outcomes, args.stake, args.sign)
    if args.attempts is not None:
        config.num_attempts = args.attempts
    if args.violation_threshold is not None:
        config.violation_threshold = args.violation_threshold
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.early is not None:
        config.early_termination_solutions = args.early
    if args.distinct_threshold is not None:
        config.distinct_threshold = args.distinct_threshold
    if args.lot_min is not None:
        config.lottery_min = args.lot_min
    if args.lot_max is not None:
        config.lottery_max = args.lot_max
    if args.min_bound is not None:
        config.lottery_min_bound = args.min_bound
    if args.max_bound is not None:
        config.lottery_max_bound = args.max_bound
    if args.save_progress:
        config.save_progress_enabled = True
    if args.alt_params:
        config.alt_params = args.alt_params
    if args.alt_workers is not None:
        config.alt_param_workers = args.alt_workers
    if args.cores is not None:
        config.num_cores = max(1, args.cores)

    seed_values = args.seed if args.seed else [None]

    if len(seed_values) > 1:
        seed_workers = args.seed_workers or min(len(seed_values), mp.cpu_count() or len(seed_values))
        base_state = asdict(config)
        if args.cores is None:
            total_cpus = mp.cpu_count() or 1
            base_state['num_cores'] = max(1, total_cpus // seed_workers)
        print(f"Running {len(seed_values)} seeds with {seed_workers} parallel processes "
              f"(cores per run: {base_state.get('num_cores')})")
        results = []
        with ProcessPoolExecutor(max_workers=seed_workers) as executor:
            futures = {
                executor.submit(run_seed_job, base_state, seed, idx): seed
                for idx, seed in enumerate(seed_values)
            }
            for future in as_completed(futures):
                res = future.result()
                results.append(res)
                print(f"[seed {res['seed']}] solutions={res['solutions_found']}, "
                      f"best={res['best_violation']}, saved={res['txt_file']}")
        return results

    seed_to_use = seed_values[0] if seed_values else None
    applied_seed, numpy_seed = apply_global_seeds(seed_to_use)
    config.seed = applied_seed

    optimizer = DistinctRPLotteryOptimizer(config)
    print(f"Starting with seed={applied_seed} (numpy_seed={numpy_seed})...")
    solutions = optimizer.solve()

    if solutions:
        print(f"\nFound {len(solutions)} solutions")
        txt = optimizer.save_solutions_to_file(solutions)
        if txt:
            print(f"Saved: {txt}")
        if config.save_progress_enabled:
            optimizer.save_progress(solutions)
        top_k = min(3 if config.outcomes == 6 else 5, len(solutions))
        print(f"\nTop {top_k} solutions:")
        for i, sol in enumerate(solutions[:top_k]):
            Z, p, Z_base = optimizer.create_lottery_structure(sol.params)
            R_vals = optimizer._collect_R_values(Z, p, sol.params)
            print(f"  {i+1}. violation={sol.violation:.6f}")
            print(f"     nominal outcomes : {Z_base.astype(int).tolist()}")
            print(f"     discounted       : {[round(float(z), 3) for z in Z]}")
            print(f"     probs            : {[round(float(q), 3) for q in p]}")
            print(f"     E[Z]             : {float(np.sum(Z * p)):.4f}")
            print(f"     reference points : {[round(r, 3) if r is not None else None for r in R_vals]}")
    else:
        print("No solutions found. Try increasing --attempts or relaxing "
              "--violation_threshold / --distinct_threshold.")


if __name__ == "__main__":
    main()
