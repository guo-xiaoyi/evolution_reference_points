# -*- coding: utf-8 -*-
"""
Unified Discrete Lottery Optimizer (4 or 6 outcomes, hi/lo stake)

This merges functionality from:
- discrete_searcher_hi_stake_four.py
- discrete_searcher_lo_stake_four.py
- discrete_searcher_hi_stake_six.py
- discrete_searcher_lo_stake_six.py

Usage examples (CLI):
  python Simulation/discrete_searcher_merged.py --outcomes 4 --stake hi
  python Simulation/discrete_searcher_merged.py --outcomes 4 --stake lo
  python Simulation/discrete_searcher_merged.py --outcomes 6 --stake hi
  python Simulation/discrete_searcher_merged.py --outcomes 6 --stake lo
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
from dataclasses import dataclass, asdict
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


def apply_global_seeds(seed: Optional[int] = None) -> Tuple[int, int]:
    seed_value = DEFAULT_RANDOM_SEED if seed is None else int(seed)
    numpy_seed = DEFAULT_NUMPY_SEED if seed is None else seed_value
    random.seed(seed_value)
    np.random.seed(numpy_seed)
    return seed_value, numpy_seed


apply_global_seeds()
warnings.filterwarnings('ignore')

# Manually specify alternate CPT parameter sets here (alpha, lambda, gamma)
# Example: ALT_PARAMS = ["0.88,2.25,0.61", "0.80,2.00,0.70"]
# alpha_candidates = np.linspace(0.95, 0.95, 1)
# lambda_candidates = np.linspace(2.25, 2.25, 1)
# gamma_candidates = np.linspace(0.61, 0.61, 1)   
alpha_candidates = np.linspace(0.75, 0.95, 5)
# longer interval
lambda_candidates = np.linspace(1.25, 2.5, 5)
# shorter interval
#lambda_candidates = np.linspace(1.6, 2.5, 5) # let lambda vary more widely (1 allows loss neutral)
gamma_candidates = np.linspace(0.58, 0.70, 5)
r_candidates = np.linspace(0.96, 0.99, 4)

ALT_PARAMS: Optional[List[str]] = [f"{alpha},{lambda_},{gamma}, {r}" for alpha in alpha_candidates for lambda_ in lambda_candidates for gamma in gamma_candidates for r in r_candidates]


# Try to import numba, fallback to regular functions if unavailable
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


# Core numerical functions with Numba JIT compilation
@jit(nopython=True, cache=True)
def fast_value_function(Z, R, alpha, lambda_):
    diff = Z - R
    result = np.empty_like(diff)
    for i in range(len(diff)):
        if diff[i] >= 0:
            result[i] = (abs(diff[i]) + 1e-12) ** alpha # gain domain
        else:
            result[i] = -lambda_ * (abs(diff[i]) + 1e-12) ** alpha # loss domain
    return result


@jit(nopython=True, cache=True)
# probability weihting function
def fast_probability_weighting(p, gamma):
    p_clipped = max(1e-10, min(p, 1 - 1e-10))
    numerator = p_clipped ** gamma
    denominator = (p_clipped ** gamma + (1 - p_clipped) ** gamma) ** (1 / gamma)
    return numerator / denominator


@jit(nopython=True, cache=True)
# compute Y function (value function)
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


@dataclass
class OptimizedConfig:
    alpha: float = 0.88
    lambda_: float = 2.25
    gamma: float = 0.61
    r : float = 0.98
    seed: Optional[int] = None

    # Mode
    outcomes: int = 4  # 4 or 6
    stake: str = "lo"  # "hi" or "lo"

    # Discrete parameter ranges (interpreted by stake/outcome presets)
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

    # Performance optimization settings
    batch_size: int = 10000
    early_termination_solutions: int = 50
    use_fast_prefilter: bool = True

    # Output settings
    output_dir: str = "lottery_results"
    save_progress_enabled: bool = True
    alt_params: Optional[List[str]] = None  # list of "alpha,lambda,gamma, r" strings
    alt_param_workers: Optional[int] = None  # threads dedicated to alternate-parameter validation

    def __post_init__(self):
        if self.prob_choices is None:
            self.prob_choices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        if self.num_cores is None:
            cpu_total = mp.cpu_count() or 1
            self.num_cores = cpu_total

        if self.outcomes == 6:
            self.output_dir = "lottery_results_6outcomes"
        os.makedirs(self.output_dir, exist_ok=True)
        # If not provided via CLI, use ALT_PARAMS constant
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
            self.probabilities = list(self.params[8:])  # p1,p2,p3
            (self.b11, self.b12, self.c21, self.c22, self.c31, self.c32, self.c33, self.c34) = self.lottery_values
            (self.p1, self.p2, self.p3) = self.probabilities
        else:
            self.lottery_values = [int(x) for x in self.params[:11]]
            self.probabilities = list(self.params[11:])  # p1..p5
            (self.b11, self.b12, self.c21, self.c22, self.c23, self.c31, self.c32, self.c33, self.c34, self.c35, self.c36) = self.lottery_values
            (self.p1, self.p2, self.p3, self.p4, self.p5) = self.probabilities

    def __str__(self):
        return f"Solution({self.outcomes}out, viol={self.violation:.6f}, vals={self.lottery_values}, probs={self.probabilities})"

    def to_dict(self):
        return {
            'outcomes': self.outcomes,
            'params': self.params.tolist(),
            'violation': self.violation,
            'timestamp': self.timestamp,
            'lottery_values': self.lottery_values,
            'probabilities': self.probabilities
        }


@dataclass
class LotteryStructure:
    Z: np.ndarray
    p: np.ndarray
    outcomes_base: np.ndarray
    expected_value: float


class UnifiedLotteryOptimizer:
    def __init__(self, config: OptimizedConfig):
        self.config = config
        self.alpha = config.alpha
        self.lambda_ = config.lambda_
        self.gamma = config.gamma
        self.r= config.r
        self.outcomes = config.outcomes
        self.stake = config.stake
        self._alt_param_specs, self._alt_param_specs_invalid = self._parse_alt_param_specs(self.config.alt_params)
        if self._alt_param_specs:
            self._alt_param_workers = min(self.config.alt_param_workers, len(self._alt_param_specs))
        else:
            self._alt_param_workers = 0

        # Statistics
        self.stats = {
            'evaluations': 0,
            'solutions_found': 0,
            'best_violation': float('inf'),
            'prefilter_rejections': 0,
            'start_time': None,
            'end_time': None,
        }

        # Build discrete space per mode
        self._prepare_discrete_space()
        self._psutil_process = psutil.Process(os.getpid()) if PSUTIL_AVAILABLE else None
        self._worker_ps = []
        self._resource_sample_interval = 1.0  # seconds between psutil polls
        self._last_resource_sample = 0.0
        self._cached_resource_status: Optional[Dict[str, str]] = None
        if self._psutil_process:
            self._psutil_process.cpu_percent(None)

    def _prepare_discrete_space(self):
        if self.outcomes == 4 and self.stake == 'hi':
            # Use separated bounds; keep multiples of 5
            low = np.arange(self.config.lottery_min_bound, self.config.lottery_min + 1)
            hi = np.arange(self.config.lottery_max, self.config.lottery_max_bound + 1)
            all_values = np.concatenate([low, hi])
            self.lottery_values = all_values[(all_values % 5 == 0)]
        elif self.outcomes == 6 and self.stake == 'hi':
            low = np.arange(self.config.lottery_min_bound, self.config.lottery_min + 1)
            hi = np.arange(self.config.lottery_max, self.config.lottery_max_bound + 1)
            all_values = np.concatenate([low, hi])
            self.lottery_values = all_values[(all_values % 5 == 0)]
        else:
            # lo stake: contiguous integer range
            self.lottery_values = np.arange(self.config.lottery_min, self.config.lottery_max + 1)

        self.prob_choices = np.array(self.config.prob_choices)

    def _parse_alt_param_specs(self, raw_specs: Optional[List[str]]) -> Tuple[List[Tuple[float, float, float, Optional[float]]], bool]:
        if not raw_specs:
            return [], False
        parsed: List[Tuple[float, float, float, Optional[float]]] = []
        for spec in raw_specs:
            try:
                parts = [x.strip() for x in spec.split(',')]
                if len(parts) not in (3, 4):
                    raise ValueError(f"Invalid alt param format: '{spec}'")
                alpha = float(parts[0])
                lambda_val = float(parts[1])
                gamma = float(parts[2])
                r_override = float(parts[3]) if len(parts) == 4 else None
                parsed.append((alpha, lambda_val, gamma, r_override))
            except Exception:
                warnings.warn(f"Invalid alternate parameter specification skipped: '{spec}'", RuntimeWarning)
                return [], True
        return parsed, False

    def _progress_desc(self):
        mode = "Numba JIT" if NUMBA_AVAILABLE else "Python"
        return f"Batch optimization [{mode}]"

    def _resource_postfix(self):
        base_status = {
            'Numba': 'On' if NUMBA_AVAILABLE else 'Off',
            'Cores': self.config.num_cores
        }
        if not self._psutil_process:
            base_status['CoreUse'] = 'n/a'
            base_status['RAM'] = 'n/a'
            return base_status

        now = time.time()
        if (
            self._cached_resource_status is not None
            and now - self._last_resource_sample < self._resource_sample_interval
        ):
            cached = dict(self._cached_resource_status)
            cached['Numba'] = base_status['Numba']
            cached['Cores'] = base_status['Cores']
            return cached

        status = dict(base_status)
        total_rss = 0
        try:
            total_rss += self._psutil_process.memory_info().rss
        except psutil.Error:
            pass
        live_workers = []
        if self._worker_ps:
            for worker_proc in self._worker_ps:
                try:
                    if worker_proc.is_running():
                        total_rss += worker_proc.memory_info().rss
                        live_workers.append(worker_proc)
                except psutil.Error:
                    continue
            self._worker_ps = live_workers
        mem_gb = total_rss / (1024 ** 3) if total_rss else 0.0
        cpu_pct = self._psutil_process.cpu_percent(None)
        total_cores = psutil.cpu_count() or 1
        core_usage = (cpu_pct / 100.0) * total_cores
        status['CoreUse'] = f"{core_usage:.1f}"
        status['RAM'] = f"{mem_gb:.2f}GB"
        if self._worker_ps:
            status['Workers'] = len(self._worker_ps)

        self._cached_resource_status = dict(status)
        self._last_resource_sample = now
        return status

    def _register_solution(self, solution: 'CompleteSolution', container: List['CompleteSolution']):
        container.append(solution)
        self.stats['solutions_found'] += 1
        if solution.violation < self.stats['best_violation']:
            self.stats['best_violation'] = solution.violation

    # ---------- Structure creation ----------
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
            return outcomes, probabilities, outcomes_base
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
            outcomes_base = np.array([z1_base, z2_base, z3_base, z4_base, z5_base, z6_base], dtype=np.int64)
            return outcomes, probabilities, outcomes_base

    # ---------- Helper computations ----------
    def _probability_weighting(self, p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        numerator = p ** self.gamma
        denominator = (p ** self.gamma + (1 - p) ** self.gamma) ** (1 / self.gamma)
        return numerator / denominator

    # Alternate-parameter versions used for sanity checks
    def _probability_weighting_params(self, p, gamma):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        numerator = p ** gamma
        denominator = (p ** gamma + (1 - p) ** gamma) ** (1 / gamma)
        return numerator / denominator

    def _compute_Y_python(self, Z, p, R, alpha, lambda_, gamma):
        diff = Z - R
        values = np.where(diff >= 0,
                          np.power(np.abs(diff) + 1e-12, alpha),
                          -lambda_ * np.power(np.abs(diff) + 1e-12, alpha))

        gain_idx = np.where(diff >= 0)[0]
        loss_idx = np.where(diff < 0)[0]
        total = 0.0

        if gain_idx.size > 0:
            gains_Z = Z[gain_idx]
            gains_p = p[gain_idx]
            gains_v = values[gain_idx]
            order = np.argsort(-gains_Z)
            gains_p_sorted = gains_p[order]
            gains_v_sorted = gains_v[order]
            cum = 0.0
            prev_w = 0.0
            for j in range(gains_p_sorted.size):
                cum += float(gains_p_sorted[j])
                w_cum = self._probability_weighting_params(cum, gamma)
                dw = w_cum - prev_w
                total += gains_v_sorted[j] * dw
                prev_w = w_cum

        if loss_idx.size > 0:
            losses_Z = Z[loss_idx]
            losses_p = p[loss_idx]
            losses_v = values[loss_idx]
            order = np.argsort(losses_Z)
            losses_p_sorted = losses_p[order]
            losses_v_sorted = losses_v[order]
            cum = 0.0
            prev_w = 0.0
            for j in range(losses_p_sorted.size):
                cum += float(losses_p_sorted[j])
                w_cum = self._probability_weighting_params(cum, gamma)
                dw = w_cum - prev_w
                total += losses_v_sorted[j] * dw
                prev_w = w_cum

        return float(total)

    def _compute_Y(self, Z, p, R):
        if NUMBA_AVAILABLE:
            return float(fast_compute_Y(np.asarray(Z, dtype=np.float64),
                                        np.asarray(p, dtype=np.float64),
                                        float(R),
                                        self.alpha,
                                        self.lambda_,
                                        self.gamma))
        return self._compute_Y_python(Z, p, R, self.alpha, self.lambda_, self.gamma)

    def _compute_Y_params(self, Z, p, R, alpha, lambda_, gamma):
        if NUMBA_AVAILABLE:
            return float(fast_compute_Y(np.asarray(Z, dtype=np.float64),
                                        np.asarray(p, dtype=np.float64),
                                        float(R),
                                        alpha,
                                        lambda_,
                                        gamma))
        return self._compute_Y_python(Z, p, R, alpha, lambda_, gamma)

    def find_R_where_Y_equals_zero(self, Z, p):
        def equation(R):
            return self._compute_Y(Z, p, R)
        for start_R in [0.0, float(np.mean(Z)), float(np.median(Z))]:
            try:
                R_solution = fsolve(equation, start_R, xtol=1e-6)[0]
                if abs(equation(R_solution)) < 1e-4:
                    return float(R_solution)
            except Exception:
                continue
        return None

    def find_R_where_Y_equals_zero_params(self, Z, p, alpha, lambda_, gamma):
        def equation(R):
            return self._compute_Y_params(Z, p, R, alpha, lambda_, gamma)
        for start_R in [0.0, float(np.mean(Z)), float(np.median(Z))]:
            try:
                R_solution = fsolve(equation, start_R, xtol=1e-6)[0]
                if abs(equation(R_solution)) < 1e-4:
                    return float(R_solution)
            except Exception:
                continue
        return None

    def find_monotonic_interval(self, Z, p):
        try:
            R_zero = self.find_R_where_Y_equals_zero(Z, p)
            if R_zero is None:
                return None, None
            larger_outcomes = Z[Z > R_zero]
            if len(larger_outcomes) == 0:
                next_reference = float(np.max(Z))
            else:
                differences = larger_outcomes - R_zero
                min_diff_idx = int(np.argmin(differences))
                next_reference = float(larger_outcomes[min_diff_idx])
            if next_reference <= R_zero:
                sorted_outcomes = np.sort(Z)
                if len(sorted_outcomes) >= 2:
                    next_reference = float(sorted_outcomes[-2])
                else:
                    next_reference = R_zero + 1.0
            return R_zero, next_reference
        except Exception:
            return None, None

    def find_monotonic_interval_params(self, Z, p, alpha, lambda_, gamma):
        try:
            R_zero = self.find_R_where_Y_equals_zero_params(Z, p, alpha, lambda_, gamma)
            if R_zero is None:
                return None, None
            larger_outcomes = Z[Z > R_zero]
            if len(larger_outcomes) == 0:
                next_reference = float(np.max(Z))
            else:
                differences = larger_outcomes - R_zero
                min_diff_idx = int(np.argmin(differences))
                next_reference = float(larger_outcomes[min_diff_idx])
            if next_reference <= R_zero:
                sorted_outcomes = np.sort(Z)
                if len(sorted_outcomes) >= 2:
                    next_reference = float(sorted_outcomes[-2])
                else:
                    next_reference = R_zero + 1.0
            return R_zero, next_reference
        except Exception:
            return None, None

    # ---------- Constraints ----------
    def _basic_constraint_violation(self, params, return_structure: bool = False):
        # probability bounds
        if self.outcomes == 4:
            p1, p2, p3 = params[8:]
            if not (0 <= p1 <= 1 and 0 <= p2 <= 1 and 0 <= p3 <= 1):
                return (True, 1000.0, None) if return_structure else (True, 1000.0)
        else:
            p1, p2, p3, p4, p5 = params[11:]
            if not (0 <= p1 <= 1 and 0 <= p2 <= 1 and 0 <= p3 <= 1 and 0 <= p4 <= 1 and 0 <= p5 <= 1):
                return (True, 1000.0, None) if return_structure else (True, 1000.0)

        Z, p, outcomes_base = self.create_lottery_structure(params)
        expected_value = float(np.sum(Z * p))
        if self.outcomes == 6 and self.stake == 'hi':
            ev_violation = (abs(expected_value)) ** 2
        elif self.outcomes == 6 and self.stake == 'lo':
            ev_violation = (abs(expected_value)) ** 2 
        elif self.outcomes == 4 and self.stake == 'hi':
            ev_violation = (abs(expected_value)) ** 2 
        else:
            ev_violation = (abs(expected_value)) ** 2 
        result = (False, float(ev_violation))
        if return_structure:
            structure = LotteryStructure(Z=Z, p=p, outcomes_base=outcomes_base, expected_value=expected_value)
            return False, float(ev_violation), structure
        return result

    def _evaluate_alt_param_spec(self, params: np.ndarray, spec: Tuple[float, float, float, Optional[float]], base_structure: Optional[LotteryStructure] = None) -> bool:
        alpha, lambda_, gamma, r_override = spec
        structure_hint = base_structure if (base_structure is not None and r_override is None) else None
        alt_viol, alt_ok = self.check_constraints_with_params(
            params,
            alpha,
            lambda_,
            gamma,
            r_override,
            base_structure=structure_hint
        )
        if not alt_ok:
            return False
        return alt_viol.get('total', float('inf')) < self.config.violation_threshold

    def _alt_param_worker_task(self, params: np.ndarray, specs: List[Tuple[float, float, float, Optional[float]]], stop_event: threading.Event, base_structure: Optional[LotteryStructure]) -> bool:
        for spec in specs:
            if stop_event.is_set():
                return True
            if not self._evaluate_alt_param_spec(params, spec, base_structure):
                stop_event.set()
                return False
        return True

    def _alt_params_valid(self, params, base_structure: Optional[LotteryStructure] = None) -> bool:
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
        chunks: List[List[Tuple[float, float, float, Optional[float]]]] = [[] for _ in range(worker_count)]
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

    def check_full_constraints(self, params, track_stats: bool = True, precomputed: Optional[LotteryStructure] = None):
        try:
            if track_stats:
                self.stats['evaluations'] += 1
            if precomputed is None:
                Z, p, outcomes_base = self.create_lottery_structure(params)
                expected_value = float(np.sum(Z * p))
                structure = LotteryStructure(Z=Z, p=p, outcomes_base=outcomes_base, expected_value=expected_value)
            else:
                structure = precomputed
                Z = structure.Z
                p = structure.p
                expected_value = structure.expected_value
            violations = {}
            total_violations = 0.0

            # Expected value weighting consistent with originals
            if self.outcomes == 6 and self.stake == 'hi':
                violations['expected_value'] = (abs(expected_value)) ** 2
            elif self.outcomes == 6 and self.stake == 'lo':
                violations['expected_value'] = (abs(expected_value)) * 5
            elif self.outcomes == 4 and self.stake == 'hi':
                violations['expected_value'] = (abs(expected_value)) * 10
            else:
                violations['expected_value'] = (abs(expected_value))
            total_violations += violations['expected_value']

            # Monotonic intervals
            IL_lower, IL_upper = self.find_monotonic_interval(Z, p)
            if IL_lower is None or IL_upper is None or IL_lower >= IL_upper:
                violations['empty_interval'] = 1000
                total_violations += 1000
                violations['total'] = total_violations
                return violations, False

            # Ensure 0 in IL
            interval_violations_L = 0.0
            if 0 < IL_lower:
                interval_violations_L += (IL_lower - 0)
            elif 0 > IL_upper:
                interval_violations_L += (0 - IL_upper)
            violations['interval_L'] = interval_violations_L
            total_violations += interval_violations_L

            # Sub-lottery intervals
            if self.outcomes == 4:
                z1, z2, z3, z4 = Z
                p1, p2, p3 = params[8:]
                Z_L1 = np.array([z1, z2]); p_L1 = np.array([p2, 1 - p2])
                Z_L2 = np.array([z3, z4]); p_L2 = np.array([p3, 1 - p3])
                E_L1 = np.sum(Z_L1 * p_L1)
                E_L2 = np.sum(Z_L2 * p_L2)
                b11, b12, c21, c22 = params[:4]

                IL1_lower, IL1_upper = self.find_monotonic_interval(Z_L1, p_L1)
                if IL1_lower is None or IL1_upper is None or IL1_lower >= IL1_upper:
                    total_violations += 100
                else:
                    for value in [b11, b11 + c21, E_L1, expected_value]:
                        if value < IL1_lower:
                            total_violations += (IL1_lower - value)
                        elif value > IL1_upper:
                            total_violations += (value - IL1_upper)

                IL2_lower, IL2_upper = self.find_monotonic_interval(Z_L2, p_L2)
                if IL2_lower is None or IL2_upper is None or IL2_lower >= IL2_upper:
                    total_violations += 100
                else:
                    for value in [b12, b12 + c22, E_L2, expected_value]:
                        if value < IL2_lower:
                            total_violations += (IL2_lower - value)
                        elif value > IL2_upper:
                            total_violations += (value - IL2_upper)
            else:
                z1, z2, z3, z4, z5, z6 = Z
                p1, p2, p3, p4, p5 = params[11:]
                Z_L1 = np.array([z1, z2]); p_L1 = np.array([p2, 1 - p2])
                Z_L2 = np.array([z3, z4]); p_L2 = np.array([p4, 1 - p4])
                Z_L3 = np.array([z5, z6]); p_L3 = np.array([p5, 1 - p5])
                p_L4 = np.array([p3 * p4, p3 * (1 - p4), (1 - p3) * p5, (1 - p3) * (1 - p5)])
                Z_L4 = np.array([z3, z4, z5, z6])
                E_L2 = np.sum(Z_L2 * p_L2)
                E_L3 = np.sum(Z_L3 * p_L3)
                E_L4 = p3 * E_L2 + (1 - p3) * E_L3
                b11, b12, c21, c22, c23 = params[:5]

                for (Z_sub, p_sub, values, penalty_key) in [
                    (Z_L1, p_L1, [b11, b11 + c21, np.sum(Z_L1 * p_L1), expected_value], 'L1'),
                    (Z_L2, p_L2, [b12, b12 + c22, E_L2, E_L4], 'L2'),
                    (Z_L3, p_L3, [b12, b12 + c23, E_L3, E_L4], 'L3'),
                ]:
                    lo, up = self.find_monotonic_interval(Z_sub, p_sub)
                    if lo is None or up is None or lo >= up:
                        total_violations += 100
                    else:
                        for value in values:
                            if value < lo:
                                total_violations += (lo - value)
                            elif value > up:
                                total_violations += (value - up)

                lo4, up4 = self.find_monotonic_interval(Z_L4, p_L4)
                if lo4 is None or up4 is None or lo4 >= up4:
                    total_violations += 100
                else:
                    for value in [b12, E_L4, expected_value]:
                        if value < lo4:
                            total_violations += (lo4 - value)
                        elif value > up4:
                            total_violations += (value - up4)

            # Probability bounds soft check
            prob_bound_violations = 0.0
            for prob in (params[8:] if self.outcomes == 4 else params[11:]):
                if prob < 0:
                    prob_bound_violations += (-prob)
                if prob > 1:
                    prob_bound_violations += (prob - 1)
            total_violations += prob_bound_violations
            violations['prob_bounds'] = prob_bound_violations

            violations['total'] = float(total_violations)
            if total_violations < self.stats['best_violation']:
                self.stats['best_violation'] = float(total_violations)
            return violations, True, structure
        except Exception as e:
            return {'total': 10000, 'error': str(e)}, False, None

    def check_constraints_with_params(self, params, alpha, lambda_, gamma, r_override: Optional[float] = None, base_structure: Optional[LotteryStructure] = None):
        try:
            if base_structure is not None and r_override is None:
                Z = base_structure.Z
                p = base_structure.p
                expected_value = base_structure.expected_value
            else:
                Z, p, _ = self.create_lottery_structure(params, r_override)
                expected_value = float(np.sum(Z * p))
            total_violations = 0.0
            # Use a moderate weighting for EV under alt params
            total_violations += float(abs(expected_value))

            # Main interval
            IL_lower, IL_upper = self.find_monotonic_interval_params(Z, p, alpha, lambda_, gamma)
            if IL_lower is None or IL_upper is None or IL_lower >= IL_upper:
                total_violations += 1000
                return {'total': total_violations, 'empty_interval': 1000}, False

            # 0 in IL
            if 0 < IL_lower:
                total_violations += (IL_lower - 0)
            elif 0 > IL_upper:
                total_violations += (0 - IL_upper)

            if self.outcomes == 4:
                z1, z2, z3, z4 = Z
                _, _, _, _, _, _, _, _, p1, p2, p3 = params
                Z_L1 = np.array([z1, z2]); p_L1 = np.array([p2, 1 - p2])
                Z_L2 = np.array([z3, z4]); p_L2 = np.array([p3, 1 - p3])
                E_L1 = np.sum(Z_L1 * p_L1)
                E_L2 = np.sum(Z_L2 * p_L2)
                b11, b12, c21, c22 = params[:4]

                lo1, up1 = self.find_monotonic_interval_params(Z_L1, p_L1, alpha, lambda_, gamma)
                if lo1 is None or up1 is None or lo1 >= up1:
                    total_violations += 100
                else:
                    for value in [b11, b11 + c21, E_L1, expected_value]:
                        if value < lo1:
                            total_violations += (lo1 - value)
                        elif value > up1:
                            total_violations += (value - up1)

                lo2, up2 = self.find_monotonic_interval_params(Z_L2, p_L2, alpha, lambda_, gamma)
                if lo2 is None or up2 is None or lo2 >= up2:
                    total_violations += 100
                else:
                    for value in [b12, b12 + c22, E_L2, expected_value]:
                        if value < lo2:
                            total_violations += (lo2 - value)
                        elif value > up2:
                            total_violations += (value - up2)
            else:
                z1, z2, z3, z4, z5, z6 = Z
                _, _, _, _, _, _, _, _, _, _, _, p1, p2, p3, p4, p5 = params
                Z_L1 = np.array([z1, z2]); p_L1 = np.array([p2, 1 - p2])
                Z_L2 = np.array([z3, z4]); p_L2 = np.array([p4, 1 - p4])
                Z_L3 = np.array([z5, z6]); p_L3 = np.array([p5, 1 - p5])
                Z_L4 = np.array([z3, z4, z5, z6])
                p_L4 = np.array([p3 * p4, p3 * (1 - p4), (1 - p3) * p5, (1 - p3) * (1 - p5)])
                E_L2 = np.sum(Z_L2 * p_L2)
                E_L3 = np.sum(Z_L3 * p_L3)
                E_L4 = p3 * E_L2 + (1 - p3) * E_L3
                b11, b12, c21, c22, c23 = params[:5]

                for (Z_sub, p_sub, values) in [
                    (Z_L1, p_L1, [b11, b11 + c21, np.sum(Z_L1 * p_L1), expected_value]),
                    (Z_L2, p_L2, [b12, b12 + c22, E_L2, E_L4]),
                    (Z_L3, p_L3, [b12, b12 + c23, E_L3, E_L4]),
                ]:
                    lo, up = self.find_monotonic_interval_params(Z_sub, p_sub, alpha, lambda_, gamma)
                    if lo is None or up is None or lo >= up:
                        total_violations += 100
                    else:
                        for value in values:
                            if value < lo:
                                total_violations += (lo - value)
                            elif value > up:
                                total_violations += (value - up)

                lo4, up4 = self.find_monotonic_interval_params(Z_L4, p_L4, alpha, lambda_, gamma)
                if lo4 is None or up4 is None or lo4 >= up4:
                    total_violations += 100
                else:
                    for value in [b12, E_L4, expected_value]:
                        if value < lo4:
                            total_violations += (lo4 - value)
                        elif value > up4:
                            total_violations += (value - up4)

            # Probability bounds soft check
            prob_bound_violations = 0.0
            for prob in (params[8:] if self.outcomes == 4 else params[11:]):
                if prob < 0:
                    prob_bound_violations += (-prob)
                if prob > 1:
                    prob_bound_violations += (prob - 1)
            total_violations += prob_bound_violations
            return {'total': float(total_violations)}, True
        except Exception as e:
            return {'total': 10000, 'error': str(e)}, False

    # ---------- Generation ----------
    def generate_batch_params(self, batch_size: int) -> np.ndarray:
        if self.outcomes == 4:
            lottery_draws = np.random.choice(self.lottery_values, size=(batch_size, 8)).astype(np.float64, copy=False)
            prob_draws = np.random.choice(self.prob_choices, size=(batch_size, 3)).astype(np.float64, copy=False)
            return np.concatenate([lottery_draws, prob_draws], axis=1)
        else:
            lottery_draws = np.random.choice(self.lottery_values, size=(batch_size, 11)).astype(np.float64, copy=False)
            prob_draws = np.random.choice(self.prob_choices, size=(batch_size, 5)).astype(np.float64, copy=False)
            return np.concatenate([lottery_draws, prob_draws], axis=1)

    def fast_prefilter(self, params_batch: np.ndarray, cache_structures: bool = False) -> Tuple[np.ndarray, Optional[List[LotteryStructure]]]:
        if not self.config.use_fast_prefilter:
            return params_batch, None
        valid_indices = []
        cached_structures: Optional[List[LotteryStructure]] = [] if cache_structures else None
        for i in range(len(params_batch)):
            params = params_batch[i]
            if cache_structures:
                invalid, violation, structure = self._basic_constraint_violation(params, return_structure=True)
            else:
                invalid, violation = self._basic_constraint_violation(params)
                structure = None
            if (not invalid) and violation < self.config.violation_threshold * 2:
                valid_indices.append(i)
                if cached_structures is not None and structure is not None:
                    cached_structures.append(structure)
            else:
                self.stats['prefilter_rejections'] += 1
        shape = (0, 11) if self.outcomes == 4 else (0, 16)
        filtered = params_batch[valid_indices] if valid_indices else np.empty(shape)
        return filtered, cached_structures

    # ---------- Optimization ----------
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
                worker_handles = []
                for proc in getattr(pool, "_pool", []):
                    try:
                        ps_proc = psutil.Process(proc.pid)
                        ps_proc.cpu_percent(None)
                        worker_handles.append(ps_proc)
                    except (psutil.Error, AttributeError):
                        continue
                self._worker_ps = worker_handles
        else:
            self._worker_ps = []

        pbar = tqdm(
            total=self.config.num_attempts,
            desc=self._progress_desc(),
            position=self.config.progress_position,
            leave=True,
            dynamic_ncols=True
        )
        early_stop = False
        try:
            while attempts_made < self.config.num_attempts and not early_stop:
                current_batch_size = min(self.config.batch_size, self.config.num_attempts - attempts_made)
                params_batch = self.generate_batch_params(current_batch_size)
                filtered_params, cached_structs = self.fast_prefilter(
                    params_batch,
                    cache_structures=not use_parallel
                )

                if use_parallel and filtered_params.size > 0:
                    rows = len(filtered_params)
                    worker_divisor = max(1, self.config.num_cores * 2)
                    block_size = max(1, rows // worker_divisor)
                    block_size = min(block_size, rows)
                    chunk_iter = (
                        filtered_params[start:start + block_size]
                        for start in range(0, rows, block_size)
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
                                if pool:
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
                    if cached_iter is not None:
                        param_iter = zip(filtered_params, cached_iter)
                    else:
                        param_iter = ((params, None) for params in filtered_params)
                    for params, cached_structure in param_iter:
                        violations, valid, structure = self.check_full_constraints(
                            params,
                            precomputed=cached_structure
                        )
                        if not (valid and violations['total'] < self.config.violation_threshold):
                            continue
                        if not self._alt_params_valid(params, structure):
                            continue
                        sol = CompleteSolution(self.outcomes, params, violations['total'])
                        self._register_solution(sol, solutions)
                        if len(solutions) >= self.config.early_termination_solutions:
                            early_stop = True
                            break

                attempts_made += current_batch_size
                pbar.update(current_batch_size)
                postfix = {
                    'Solutions': len(solutions),
                    'Best': f"{self.stats['best_violation']:.3f}",
                    'Prefilter': f"{self.stats['prefilter_rejections']}"
                }
                postfix.update(self._resource_postfix())
                pbar.set_postfix(postfix)
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
        print(f"Starting lottery parameter optimization...")
        print(f"Mode: outcomes={self.outcomes}, stake={self.stake}")
        print(f"Numba acceleration: {'Enabled' if NUMBA_AVAILABLE else 'Disabled'}")
        print(f"Alternate parameter set numbers: {len(self.config.alt_params)}")
        print(f"Prospect theory parameters: alpha={self.alpha}, lambda={self.lambda_}, gamma={self.gamma}, r={self.r}")
        solutions = self.batch_optimize()
        solutions.sort(key=lambda s: s.violation)
        elapsed_time = (self.stats['end_time'] or time.time()) - self.stats['start_time']
        print(f"\nOptimization completed ({elapsed_time:.2f}s)")
        print(f"Statistics:")
        print(f"   Total evaluations: {self.stats['evaluations']:,}")
        print(f"   Solutions found: {len(solutions)}")
        print(f"   Prefilter rejections: {self.stats['prefilter_rejections']:,}")
        if elapsed_time > 0:
            print(f"   Evaluation speed: {self.stats['evaluations'] / elapsed_time:.0f} evals/s")
        if solutions:
            print(f"   Best violation value: {solutions[0].violation:.6f}")
        return solutions

    # ---------- Reporting ----------
    def calculate_solution_diversity(self, solutions: List[CompleteSolution]) -> Dict[str, float]:
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
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        seed_suffix = f"_seed{self.config.seed}" if getattr(self.config, 'seed', None) is not None else ""
        if self.outcomes == 4:
            return os.path.join(self.config.output_dir, f"lottery_solutions_four{seed_suffix}_{timestamp}.txt")
        suffix = "_hi" if self.stake == 'hi' else ""
        return os.path.join(self.config.output_dir, f"lottery_6outcomes{seed_suffix}_{timestamp}{suffix}.txt")

    def save_solutions_to_file(self, solutions: List[CompleteSolution], filename: str = None) -> str:
        if filename is None:
            filename = self._default_filename()
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                title = "6 Outcomes Results" if self.outcomes == 6 else "Results Report"
                f.write(f"Unified Discrete Lottery Optimizer - {title}\n")
                f.write("=" * 80 + "\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Mode: outcomes={self.outcomes}, stake={self.stake}\n")
                f.write(f"Solutions found: {len(solutions)}\n")
                f.write(f"Prospect theory parameters: alpha={self.alpha}, lambda={self.lambda_}, gamma={self.gamma}, r={self.r}\n")
                f.write(f"Search configuration:\n")
                if self.stake == 'hi':
                    f.write(f"  - Lottery range: [{self.config.lottery_min_bound}, {self.config.lottery_min}] and [{self.config.lottery_max}, {self.config.lottery_max_bound}]\n")
                else:
                    f.write(f"  - Lottery range: [{self.config.lottery_min}, {self.config.lottery_max}]\n")
                f.write(f"  - Probability choices: {self.config.prob_choices}\n")
                f.write(f"  - Violation threshold: {self.config.violation_threshold}\n")
                f.write(f"  - Total evaluations: {self.stats['evaluations']:,}\n")
                f.write(f"  - Numba acceleration: {'Enabled' if NUMBA_AVAILABLE else 'Disabled'}\n")
                f.write("\n")

                if not solutions:
                    f.write("No solutions found!\n")
                    f.write("Suggestions:\n")
                    f.write("  - Increase num_attempts\n")
                    f.write("  - Relax violation_threshold\n")
                    f.write("  - Adjust parameter ranges\n")
                    return filename

                # Details per solution
                for i, solution in enumerate(solutions):
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Solution {i+1}\n")
                    f.write(f"{'='*60}\n")
                    f.write(f"Objective function value: {solution.violation:.6f}\n")
                    f.write(f"Timestamp: {datetime.fromtimestamp(solution.timestamp).strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("\n")

                    if self.outcomes == 4:
                        f.write("Lottery structure (4 outcomes):\n")
                        f.write(f"  Stage 1: b11={solution.b11:3d}, b12={solution.b12:3d}\n")
                        f.write(f"  Stage 2: c21={solution.c21:3d}, c22={solution.c22:3d}\n")
                        f.write(f"  Stage 3: c31={solution.c31:3d}, c32={solution.c32:3d}, c33={solution.c33:3d}, c34={solution.c34:3d}\n")
                        f.write(f"  Probabilities: p1={solution.p1:.3f}, p2={solution.p2:.3f}, p3={solution.p3:.3f}\n")
                        Z, p, Z_base = self.create_lottery_structure(solution.params)
                        z1, z2, z3, z4 = Z
                        expected_value = np.sum(Z * p)
                        f.write("Final outcomes (nominal -> discounted):\n")
                        for idx, (z_nom, z_disc, prob) in enumerate(zip(Z_base, Z, p), start=1):
                            f.write(f"  z{idx}_nom={int(z_nom):5d} -> z{idx}_disc={z_disc:.3f} (prob={prob:.3f})\n")
                        f.write(f"  Discounted expected value: {expected_value:.6f}\n")

                        # Constraint verification (detailed like example)
                        f.write("\nConstraint verification:\n")
                        f.write(f"  1. Expected value constraint: {expected_value:.6f} ≈ 0 {'✓' if abs(expected_value) < 1 else '✗'}\n")
                        f.write(f"  3. Probability constraint: all probabilities ∈ [0,1] {'✓' if all(0 <= prob <= 1 for prob in solution.probabilities) else '✗'}\n")

                        IL_lower, IL_upper = self.find_monotonic_interval(Z, p)
                        if IL_lower is not None and IL_upper is not None:
                            f.write(f"  4. Main interval IL = [{IL_lower:.3f}, {IL_upper:.3f}]\n")

                        Z_L1 = np.array([z1, z2]); p_L1 = np.array([solution.p2, 1 - solution.p2])
                        E_L1 = np.sum(Z_L1 * p_L1)
                        IL1_lower, IL1_upper = self.find_monotonic_interval(Z_L1, p_L1)
                        if IL1_lower is not None and IL1_upper is not None:
                            f.write(f"  5. L1 interval IL1 = [{IL1_lower:.3f}, {IL1_upper:.3f}]\n")
                            values_to_check_L1 = [solution.b11, solution.b11 + solution.c21, E_L1, expected_value]
                            value_names_L1 = ['b11', 'b11+c21', 'E(L1)', 'E(L)']
                            for j, value in enumerate(values_to_check_L1):
                                in_interval = (IL1_lower <= value <= IL1_upper)
                                f.write(f"     {'✓' if in_interval else '✗'} {value_names_L1[j]} = {value:.3f} {'∈' if in_interval else '∉'} IL1\n")

                        Z_L2 = np.array([z3, z4]); p_L2 = np.array([solution.p3, 1 - solution.p3])
                        E_L2 = np.sum(Z_L2 * p_L2)
                        IL2_lower, IL2_upper = self.find_monotonic_interval(Z_L2, p_L2)
                        if IL2_lower is not None and IL2_upper is not None:
                            f.write(f"  6. L2 interval IL2 = [{IL2_lower:.3f}, {IL2_upper:.3f}]\n")
                            values_to_check_L2 = [solution.b12, solution.b12 + solution.c22, E_L2, expected_value]
                            value_names_L2 = ['b12', 'b12+c22', 'E(L2)', 'E(L)']
                            for j, value in enumerate(values_to_check_L2):
                                in_interval = (IL2_lower <= value <= IL2_upper)
                                f.write(f"     {'✓' if in_interval else '✗'} {value_names_L2[j]} = {value:.3f} {'∈' if in_interval else '∉'} IL2\n")
                    else:
                        f.write("Lottery structure (6 outcomes):\n")
                        f.write(f"  Stage 1: b11={solution.b11:3d}, b12={solution.b12:3d}\n")
                        f.write(f"  Stage 2: c21={solution.c21:3d}, c22={solution.c22:3d}, c23={solution.c23:3d}\n")
                        f.write(f"  Stage 3: c31={solution.c31:3d}, c32={solution.c32:3d}, c33={solution.c33:3d}, c34={solution.c34:3d}, c35={solution.c35:3d}, c36={solution.c36:3d}\n")
                        f.write(f"  Probabilities: p1={solution.p1:.3f}, p2={solution.p2:.3f}, p3={solution.p3:.3f}, p4={solution.p4:.3f}, p5={solution.p5:.3f}\n")
                        Z, p, Z_base = self.create_lottery_structure(solution.params)
                        z1, z2, z3, z4, z5, z6 = Z
                        expected_value = np.sum(Z * p)
                        f.write("Final outcomes (nominal -> discounted):\n")
                        for idx, (z_nom, z_disc, prob) in enumerate(zip(Z_base, Z, p), start=1):
                            f.write(f"  z{idx}_nom={int(z_nom):5d} -> z{idx}_disc={z_disc:.3f} (prob={prob:.3f})\n")
                        f.write(f"  Discounted expected value: {expected_value:.6f}\n")
                        f.write(f"  Probability sum: {np.sum(p):.6f}\n")

                        # Constraint verification for 6 outcomes (detailed)
                        f.write("\nConstraint verification:\n")
                        f.write(f"  1. Discounted value constraint: {expected_value:.6f} ≈ 0 {'✓' if abs(expected_value) < 0.5 else '✗'}\n")
                        f.write(f"  3. Probability constraint: all probabilities ∈ [0,1] {'✓' if all(0 <= prob <= 1 for prob in solution.probabilities) else '✗'}\n")

                        IL_lower, IL_upper = self.find_monotonic_interval(Z, p)
                        if IL_lower is not None and IL_upper is not None:
                            f.write(f"  4. Main interval IL = [{IL_lower:.3f}, {IL_upper:.3f}]\n")
                            f.write(f"     0 ∈ IL: {IL_lower <= 0 <= IL_upper}\n")

                        # Sub-lottery expected values (all discounted)
                        E_L1 = z1 * solution.p2 + z2 * (1 - solution.p2)
                        E_L2 = z3 * solution.p4 + z4 * (1 - solution.p4)
                        E_L3 = z5 * solution.p5 + z6 * (1 - solution.p5)
                        E_L4 = solution.p3 * E_L2 + (1 - solution.p3) * E_L3

                        # Intervals and membership checks
                        Z_L1 = np.array([z1, z2]); p_L1 = np.array([solution.p2, 1 - solution.p2])
                        IL1_lower, IL1_upper = self.find_monotonic_interval(Z_L1, p_L1)
                        if IL1_lower is not None and IL1_upper is not None:
                            f.write(f"  5. L1 interval IL1 = [{IL1_lower:.3f}, {IL1_upper:.3f}]\n")
                            values_L1 = [solution.b11, solution.b11 + solution.c21, E_L1, expected_value]
                            names_L1 = ['b11', 'b11+c21', 'E(L1)', 'E(L)']
                            for j, value in enumerate(values_L1):
                                in_interval = (IL1_lower <= value <= IL1_upper)
                                f.write(f"     {'✓' if in_interval else '✗'} {names_L1[j]} = {value:.3f} {'∈' if in_interval else '∉'} IL1\n")

                        Z_L2 = np.array([z3, z4]); p_L2 = np.array([solution.p4, 1 - solution.p4])
                        IL2_lower, IL2_upper = self.find_monotonic_interval(Z_L2, p_L2)
                        if IL2_lower is not None and IL2_upper is not None:
                            f.write(f"  6. L2 interval IL2 = [{IL2_lower:.3f}, {IL2_upper:.3f}]\n")
                            values_L2 = [solution.b12, solution.b12 + solution.c22, E_L2, E_L4]
                            names_L2 = ['b12', 'b12+c22', 'E(L2)', 'E(L4)']
                            for j, value in enumerate(values_L2):
                                in_interval = (IL2_lower <= value <= IL2_upper)
                                f.write(f"     {'✓' if in_interval else '✗'} {names_L2[j]} = {value:.3f} {'∈' if in_interval else '∉'} IL2\n")

                        Z_L3 = np.array([z5, z6]); p_L3 = np.array([solution.p5, 1 - solution.p5])
                        IL3_lower, IL3_upper = self.find_monotonic_interval(Z_L3, p_L3)
                        if IL3_lower is not None and IL3_upper is not None:
                            f.write(f"  7. L3 interval IL3 = [{IL3_lower:.3f}, {IL3_upper:.3f}]\n")
                            values_L3 = [solution.b12, solution.b12 + solution.c23, E_L3, E_L4]
                            names_L3 = ['b12', 'b12+c23', 'E(L3)', 'E(L4)']
                            for j, value in enumerate(values_L3):
                                in_interval = (IL3_lower <= value <= IL3_upper)
                                f.write(f"     {'✓' if in_interval else '✗'} {names_L3[j]} = {value:.3f} {'∈' if in_interval else '∉'} IL3\n")

                        Z_L4 = np.array([z3, z4, z5, z6])
                        p_L4 = np.array([solution.p3 * solution.p4, solution.p3 * (1 - solution.p4), (1 - solution.p3) * solution.p5, (1 - solution.p3) * (1 - solution.p5)])
                        IL4_lower, IL4_upper = self.find_monotonic_interval(Z_L4, p_L4)
                        if IL4_lower is not None and IL4_upper is not None:
                            f.write(f"  8. L4 interval IL4 = [{IL4_lower:.3f}, {IL4_upper:.3f}]\n")
                            values_L4 = [solution.b12, E_L4, expected_value]
                            names_L4 = ['b12', 'E(L4)', 'E(L)']
                            for j, value in enumerate(values_L4):
                                in_interval = (IL4_lower <= value <= IL4_upper)
                                f.write(f"     {'✓' if in_interval else '✗'} {names_L4[j]} = {value:.3f} {'∈' if in_interval else '∉'} IL4\n")

                    # Alternate parameter sanity checks
                    if self.config.alt_params:
                        f.write("\nAlternate-parameter sanity checks:\n")
                        for s_idx, param_str in enumerate(self.config.alt_params):
                            try:
                                parts = [x.strip() for x in param_str.split(',')]
                                if len(parts) not in (3, 4):
                                    raise ValueError("wrong length")
                                a_alt = float(parts[0]); l_alt = float(parts[1]); g_alt = float(parts[2])
                                r_alt = float(parts[3]) if len(parts) == 4 else None
                            except Exception:
                                f.write(f"  {s_idx+1}) Invalid alt params: {param_str}\n")
                                continue
                            alt_viol, alt_ok = self.check_constraints_with_params(solution.params, a_alt, l_alt, g_alt, r_alt)
                            ok_flag = (alt_ok and alt_viol.get('total', 1e9) < self.config.violation_threshold)
                            #f.write(f"  {s_idx+1}) alpha={a_alt}, lambda={l_alt}, gamma={g_alt} -> total violation={alt_viol.get('total', 1e9):.6f} {'✓' if ok_flag else '✗'}\n")

                # Summary table
                f.write(f"\n{'='*80}\n")
                f.write("Solution Summary Table\n")
                f.write(f"{'='*80}\n")
                summary_data = []
                for i, sol in enumerate(solutions):
                    Z, p, Z_base = self.create_lottery_structure(sol.params)
                    row = {'Sol': i+1}
                    if self.outcomes == 4:
                        row.update({'b11': sol.b11, 'b12': sol.b12,
                                    'c21': sol.c21, 'c22': sol.c22,
                                    'c31': sol.c31, 'c32': sol.c32, 'c33': sol.c33, 'c34': sol.c34,
                                    'p1': sol.p1, 'p2': sol.p2, 'p3': sol.p3,
                                    'z1': int(Z_base[0]), 'z2': int(Z_base[1]),
                                    'z3': int(Z_base[2]), 'z4': int(Z_base[3]),
                                    'z1_disc': float(Z[0]), 'z2_disc': float(Z[1]),
                                    'z3_disc': float(Z[2]), 'z4_disc': float(Z[3])})
                    else:
                        row.update({'b11': sol.b11, 'b12': sol.b12,
                                    'c21': sol.c21, 'c22': sol.c22, 'c23': sol.c23,
                                    'c31': sol.c31, 'c32': sol.c32, 'c33': sol.c33, 'c34': sol.c34, 'c35': sol.c35, 'c36': sol.c36,
                                    'p1': sol.p1, 'p2': sol.p2, 'p3': sol.p3, 'p4': sol.p4, 'p5': sol.p5,
                                    'z1': int(Z_base[0]), 'z2': int(Z_base[1]), 'z3': int(Z_base[2]),
                                    'z4': int(Z_base[3]), 'z5': int(Z_base[4]), 'z6': int(Z_base[5]),
                                    'z1_disc': float(Z[0]), 'z2_disc': float(Z[1]), 'z3_disc': float(Z[2]),
                                    'z4_disc': float(Z[3]), 'z5_disc': float(Z[4]), 'z6_disc': float(Z[5])})
                    row['E[Z]'] = float(np.sum(Z * p))
                    row['Violation'] = sol.violation
                    summary_data.append(row)
                df = pd.DataFrame(summary_data)
                f.write(df.to_string(index=False, float_format='%.3f'))
                f.write("\n")

                # Diversity metrics
                diversity = self.calculate_solution_diversity(solutions)
                f.write(f"\n{'='*80}\n")
                f.write("Solution Diversity Metrics\n")
                f.write(f"{'='*80}\n")
                f.write(f"Total solutions: {diversity['num_solutions']}\n")
                if len(solutions) >= 2:
                    f.write(f"Unique lottery combinations: {diversity['unique_lottery_combinations']}\n")
                    f.write(f"Unique probability combinations: {diversity['unique_prob_combinations']}\n")
                    f.write(f"Mean pairwise distance: {diversity.get('mean_distance', 0):.3f}\n")
                    f.write(f"Distance range: [{diversity.get('min_distance', 0):.3f}, {diversity.get('max_distance', 0):.3f}]\n")
                    f.write(f"Distance standard deviation: {diversity.get('std_distance', 0):.3f}\n")

                # Performance statistics
                f.write(f"\n{'='*80}\n")
                f.write("Performance Statistics\n")
                f.write(f"{'='*80}\n")
                elapsed_time = (self.stats['end_time'] or time.time()) - self.stats['start_time']
                f.write(f"Total runtime: {elapsed_time:.2f} seconds\n")
                f.write(f"Total evaluations: {self.stats['evaluations']:,}\n")
                f.write(f"Solutions found: {self.stats['solutions_found']}\n")
                if elapsed_time > 0:
                    f.write(f"Evaluation speed: {self.stats['evaluations'] / elapsed_time:.0f} evals/s\n")
                if solutions:
                    violations = [sol.violation for sol in solutions]
                    f.write(f"Best violation value: {min(violations):.6f}\n")
                    f.write(f"Average violation value: {np.mean(violations):.6f}\n")
                    f.write(f"Worst violation value: {max(violations):.6f}\n")
            return filename
        except Exception as e:
            print(f"Error saving file: {e}")
            return None

    def save_progress(self, solutions: List[CompleteSolution], filename: str = None):
        if not self.config.save_progress_enabled:
            return None
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            suffix = f"_6outcomes" if self.outcomes == 6 else ""
            seed_suffix = f"_seed{self.config.seed}" if getattr(self.config, 'seed', None) is not None else ""
            filename = os.path.join(self.config.output_dir, f"lottery_progress{suffix}{seed_suffix}_{timestamp}.pkl")
        progress_data = {
            'solutions': [s.to_dict() for s in solutions],
            'stats': self.stats,
            'config': self.config,
            'timestamp': datetime.now()
        }
        try:
            with open(filename, 'wb') as f:
                pickle.dump(progress_data, f)
            print(f"Progress saved to: {filename}")
            return filename
        except Exception as e:
            print(f"Error saving progress: {e}")
            return None


MP_WORKER_OPTIMIZER = None


def _mp_worker_initializer(config_state: Dict):
    state_copy = dict(config_state)
    state_copy['num_cores'] = 1
    config = OptimizedConfig(**state_copy)
    global MP_WORKER_OPTIMIZER
    MP_WORKER_OPTIMIZER = UnifiedLotteryOptimizer(config)


def _mp_worker_process(params_block):
    optimizer = MP_WORKER_OPTIMIZER
    if optimizer is None:
        return {'error': 'Worker not initialized'}
    evaluations = 0
    solutions: List[Tuple[float, np.ndarray]] = []
    try:
        for params in params_block:
            evaluations += 1
            violations, valid, structure = optimizer.check_full_constraints(params, track_stats=False)
            if not (valid and violations['total'] < optimizer.config.violation_threshold):
                continue
            if not optimizer._alt_params_valid(params, structure):
                continue
            solutions.append((float(violations['total']), params.copy()))
        return {'evaluations': evaluations, 'solutions': solutions}
    except Exception as exc:
        return {'error': str(exc)}


def build_preset_config(outcomes: int, stake: str) -> OptimizedConfig:
    if outcomes == 4 and stake == 'hi':
        return OptimizedConfig(
            outcomes=4, stake='hi',
            lottery_min=-100, lottery_max=100,
            lottery_min_bound=-1000, lottery_max_bound=1000,
            num_attempts=10000000, violation_threshold=1.0,
            batch_size=10000, early_termination_solutions=25,
            output_dir="lottery_results"
        )
    if outcomes == 4 and stake == 'lo':
        return OptimizedConfig(
            outcomes=4, stake='lo',
            lottery_min=-50, lottery_max=50,
            num_attempts=10000000, violation_threshold=1.0,
            batch_size=10000, early_termination_solutions=25,
            output_dir="lottery_results"
        )
    if outcomes == 6 and stake == 'hi':
        return OptimizedConfig(
            outcomes=6, stake='hi',
            lottery_min=-500, lottery_max=500,
            lottery_min_bound=-1000, lottery_max_bound=1000,
            num_attempts=10000000, violation_threshold=1.0,
            batch_size=30000, early_termination_solutions=25,
            output_dir="lottery_results_6outcomes"
        )
    # outcomes == 6 and stake == 'lo'
    return OptimizedConfig(
        outcomes=6, stake='lo',
        lottery_min=-50, lottery_max=50,
        num_attempts=10000000, violation_threshold=1.0,
        batch_size=10000, early_termination_solutions=25,
        output_dir="lottery_results_6outcomes"
    )


def run_seed_job(config_state: Dict, seed_value: Optional[int], position: int = 0):
    applied_seed, numpy_seed = apply_global_seeds(seed_value)
    state_copy = dict(config_state)
    state_copy['seed'] = applied_seed
    state_copy['progress_position'] = position
    config = OptimizedConfig(**state_copy)
    optimizer = UnifiedLotteryOptimizer(config)
    solutions = optimizer.solve()
    txt_file = optimizer.save_solutions_to_file(solutions)
    progress_file = None
    if optimizer.config.save_progress_enabled:
        progress_file = optimizer.save_progress(solutions)
    best_violation = solutions[0].violation if solutions else None
    return {
        'seed': applied_seed,
        'numpy_seed': numpy_seed,
        'solutions_found': len(solutions),
        'best_violation': best_violation,
        'txt_file': txt_file,
        'progress_file': progress_file
    }


def main():
    parser = argparse.ArgumentParser(description="Unified Discrete Lottery Optimizer")
    parser.add_argument("--outcomes", type=int, choices=[4, 6], default=4)
    parser.add_argument("--stake", type=str, choices=["hi", "lo"], default="lo")
    parser.add_argument("--attempts", type=int, default=None)
    parser.add_argument("--violation_threshold", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--early", type=int, default=None, help="early termination solutions")
    parser.add_argument("--min", dest="lot_min", type=int, default=None)
    parser.add_argument("--max", dest="lot_max", type=int, default=None)
    parser.add_argument("--min_bound", type=int, default=None)
    parser.add_argument("--max_bound", type=int, default=None)
    parser.add_argument("--save_progress", action="store_true")
    parser.add_argument("--alt_params", nargs='*', default=None, help="Alternate parameter sets as 'alpha,lambda,gamma[,r]' (space-separated for multiple)")
    parser.add_argument("--alt_workers", type=int, default=None, help="Threads to use when validating alternate parameter sets")
    parser.add_argument("--seed", type=int, nargs='*', default=None, help="Override RNG seed(s). Provide multiple seeds to fan out runs.")
    parser.add_argument("--seed_workers", type=int, default=None, help="Parallel processes to launch when multiple seeds are provided.")
    parser.add_argument("--cores", type=int, default=None, help="CPU cores to use inside each optimization run.")
    args = parser.parse_args()

    config = build_preset_config(args.outcomes, args.stake)
    if args.attempts is not None:
        config.num_attempts = args.attempts
    if args.violation_threshold is not None:
        config.violation_threshold = args.violation_threshold
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.early is not None:
        config.early_termination_solutions = args.early
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
        base_config_state = asdict(config)
        if args.cores is None:
            total_cpus = mp.cpu_count() or 1
            base_config_state['num_cores'] = max(1, total_cpus // seed_workers)
        print(f"Running {len(seed_values)} seeds with up to {seed_workers} parallel processes (cores per run: {base_config_state.get('num_cores')})")
        results = []
        with ProcessPoolExecutor(max_workers=seed_workers) as executor:
            futures = {
                executor.submit(run_seed_job, base_config_state, seed, idx): seed
                for idx, seed in enumerate(seed_values)
            }
            for future in as_completed(futures):
                res = future.result()
                results.append(res)
                print(f"[seed {res['seed']}] solutions={res['solutions_found']}, best={res['best_violation']}, saved={res['txt_file']}")
        return results

    seed_to_use = seed_values[0] if seed_values else None
    applied_seed, numpy_seed = apply_global_seeds(seed_to_use)
    config.seed = applied_seed

    # Create optimizer
    optimizer = UnifiedLotteryOptimizer(config)

    # Run optimization
    print(f"Starting optimization with seed={applied_seed} (numpy_seed={numpy_seed})...")
    solutions = optimizer.solve()

    # Save results to file
    if solutions:
        print(f"\nFound {len(solutions)} solutions")
        txt_filename = optimizer.save_solutions_to_file(solutions)
        if txt_filename:
            print(f"Detailed results saved to: {txt_filename}")
        # Automatically persist progress instead of prompting
        if optimizer.config.save_progress_enabled:
            try:
                optimizer.save_progress(solutions)
                print("Progress file saved automatically.")
            except Exception as err:
                print(f"Failed to save progress automatically: {err}")
        # Display a few solutions
        top_k = 3 if config.outcomes == 6 else 5
        print(f"\nTop {min(top_k, len(solutions))} best solutions:")
        for i, sol in enumerate(solutions[:top_k]):
            Z, p, Z_base = optimizer.create_lottery_structure(sol.params)
            print(f"  {i+1}. Violation: {sol.violation:.6f}")
            print(f"      Outcomes nominal: {Z_base.astype(int).tolist()}")
            print(f"      Outcomes discounted: {[round(float(z), 3) for z in Z]}")
            prob_list = [f"{prob:.3f}" for prob in p]
            print(f"      Probabilities: p={prob_list}")
            print(f"      Discounted expected value: {np.sum(Z * p):.6f}")
        print("\nCompleted! See generated txt file for details.")
    else:
        print("\nNo solutions found")
        print("Suggestions:")
        print("  - Increase num_attempts")
        print("  - Relax violation_threshold")
        print("  - Adjust parameter ranges")
        txt_filename = optimizer.save_solutions_to_file([])
        if txt_filename:
            print(f"Search record saved to: {txt_filename}")
    return solutions


if __name__ == "__main__":
    solutions = main()
