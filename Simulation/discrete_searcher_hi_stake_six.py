# -*- coding: utf-8 -*-
"""
High-Performance Discrete Lottery Optimizer with File Output (For 6 Outcomes) - CORRECTED

Features:
1. Numba JIT compilation for core computations
2. Intelligent prefiltering and batch processing
3. Complete constraint checking (consistent with original version)
4. Detailed results output to txt files
5. Progress saving and loading functionality

@author: Performance Optimized Complete Version (6 Outcomes Corrected)
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
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Optional, Dict, Any
import random
from dataclasses import dataclass
from functools import lru_cache
import pickle
from scipy.spatial.distance import pdist

# Try to import numba, fallback to regular functions if unavailable
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("Please install numba to have full performance")
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

warnings.filterwarnings('ignore')


# Core numerical functions with Numba JIT compilation
@jit(nopython=True, cache=True)
def fast_value_function(Z, R, alpha, lambda_):
    """JIT compiled value function"""
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
    """JIT compiled probability weighting function"""
    p_clipped = max(1e-10, min(p, 1 - 1e-10))
    numerator = p_clipped ** gamma
    denominator = (p_clipped ** gamma + (1 - p_clipped) ** gamma) ** (1 / gamma)
    return numerator / denominator


@jit(nopython=True, cache=True)
def fast_compute_Y(Z, p, R, alpha, lambda_, gamma):
    """JIT compiled Y value computation"""
    v = fast_value_function(Z, R, alpha, lambda_)
    w = np.empty_like(p)
    for i in range(len(p)):
        w[i] = fast_probability_weighting(p[i], gamma)
    return np.sum(v * w)


@jit(nopython=True, cache=True)
def fast_create_lottery_structure(params):
    """JIT compiled lottery structure creation for 6 outcomes"""
    # CORRECTED: 11 lottery values + 5 probabilities = 16 parameters
    b11, b12, c21, c22, c23, c31, c32, c33, c34, c35, c36, p1, p2, p3, p4, p5 = params
    
    # Calculate final outcomes - CORRECTED structure
    z1 = b11 + c21 + c31  # Upper branch, first sub-branch, first outcome
    z2 = b11 + c21 + c32  # Upper branch, first sub-branch, second outcome
    z3 = b12 + c22 + c33  # Lower branch, first sub-branch, first outcome
    z4 = b12 + c22 + c34  # Lower branch, first sub-branch, second outcome
    z5 = b12 + c23 + c35  # Lower branch, second sub-branch, first outcome
    z6 = b12 + c23 + c36  # Lower branch, second sub-branch, second outcome
    
    # Calculate path probabilities - CORRECTED
    prob1 = p1 * p2                    # P(upper) * P(first sub | upper)
    prob2 = p1 * (1 - p2)              # P(upper) * P(second sub | upper)
    prob3 = (1 - p1) * p3 * p4         # P(lower) * P(first sub | lower) * P(first outcome | first sub)
    prob4 = (1 - p1) * p3 * (1 - p4)   # P(lower) * P(first sub | lower) * P(second outcome | first sub)
    prob5 = (1 - p1) * (1 - p3) * p5   # P(lower) * P(second sub | lower) * P(first outcome | second sub)
    prob6 = (1 - p1) * (1 - p3) * (1 - p5)  # P(lower) * P(second sub | lower) * P(second outcome | second sub)
    
    outcomes = np.array([z1, z2, z3, z4, z5, z6])
    probabilities = np.array([prob1, prob2, prob3, prob4, prob5, prob6])
    
    return outcomes, probabilities


@jit(nopython=True, cache=True)
def fast_check_basic_constraints(params):
    """JIT compiled basic constraint checking for 6 outcomes"""
    b11, b12, c21, c22, c23, c31, c32, c33, c34, c35, c36, p1, p2, p3, p4, p5 = params
    
    # Probability bounds check
    if p1 < 0 or p1 > 1 or p2 < 0 or p2 > 1 or p3 < 0 or p3 > 1 or p4 < 0 or p4 > 1 or p5 < 0 or p5 > 1:
        return False, 1000.0
    
    # Create lottery structure
    Z, p_array = fast_create_lottery_structure(params)
    z1, z2, z3, z4, z5, z6 = Z
    
    # Expected value constraint
    expected_value = np.sum(Z * p_array)
    expected_violation = (abs(expected_value))**2
    
    # Ordering constraints - CORRECTED for 6 outcomes
    ordering_violations = 0.0
    if z1 <= z2:
        ordering_violations += (z2 - z1 + 1)
    if z2 <= 0:
        ordering_violations += (-z2 + 1)
    if z3 <= 0:
        ordering_violations += (-z3 + 1)
    if z3 <= z4:
        ordering_violations += (z4 - z3 + 1)
    if z5 <= 0:
        ordering_violations += (-z5 + 1)
    if z5 <= z6:
        ordering_violations += (z6 - z5 + 1)
    
    # Additional ordering constraints to ensure proper structure
    if abs(z2 - z3) < 0.01:  # z2 ≠ z3
        ordering_violations += 2
    if abs(z4 - z5) < 0.01:  # z4 ≠ z5
        ordering_violations += 2
    
    total_violation = expected_violation + ordering_violations
    
    return True, total_violation


@dataclass
class OptimizedConfig:
    """Optimization configuration class"""
    alpha: float = 0.88
    lambda_: float = 2.25
    gamma: float = 0.61
    
    # Discrete parameter ranges
    lottery_min_bound: int = -101
    lottery_min: int = -26
    lottery_max: int = 26
    lottery_max_bound: int = 101
    prob_choices: List[float] = None
    
    # Optimization settings
    num_attempts: int = 1000000
    violation_threshold: float = 5.0
    num_cores: Optional[int] = None
    
    # Performance optimization settings
    batch_size: int = 1000
    early_termination_solutions: int = 50
    use_fast_prefilter: bool = True
    cache_size: int = 10000
    
    # Output settings
    output_dir: str = "lottery_results"
    save_progress: bool = True
    
    def __post_init__(self):
        if self.prob_choices is None:
            self.prob_choices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        if self.num_cores is None:
            self.num_cores = min(mp.cpu_count(), 8)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)


class CompleteSolution:
    """Complete solution class for 6 outcomes"""
    
    def __init__(self, params: np.ndarray, violation: float, timestamp: float = None):
        self.params = params.copy()
        self.violation = violation
        self.timestamp = timestamp or time.time()
        
        # Extract components - CORRECTED for 16 parameters
        self.lottery_values = [int(x) for x in self.params[:11]]  # 11 lottery values
        self.probabilities = list(self.params[11:])  # 5 probabilities
        
        # Calculate additional information - CORRECTED assignment
        self.b11, self.b12, self.c21, self.c22, self.c23, self.c31, self.c32, self.c33, self.c34, self.c35, self.c36 = self.lottery_values
        self.p1, self.p2, self.p3, self.p4, self.p5 = self.probabilities
    
    def __str__(self):
        return f"Solution(violation={self.violation:.6f}, lottery={self.lottery_values}, probs={self.probabilities})"
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'params': self.params.tolist(),
            'violation': self.violation,
            'timestamp': self.timestamp,
            'lottery_values': self.lottery_values,
            'probabilities': self.probabilities
        }


class OptimizedLotteryOptimizer:
    """High-performance complete lottery optimizer for 6 outcomes"""
    
    def __init__(self, config: OptimizedConfig = None):
        self.config = config or OptimizedConfig()
        self.alpha = self.config.alpha
        self.lambda_ = self.config.lambda_
        self.gamma = self.config.gamma
        
        # Pre-compute discrete space
        self.lottery_values_low_bound = np.arange(self.config.lottery_min_bound, self.config.lottery_min + 1)
        self.lottery_values_hi_bound = np.arange(self.config.lottery_max, self.config.lottery_max_bound + 1)
        self.lottery_values = np.concatenate([self.lottery_values_low_bound, self.lottery_values_hi_bound])
        self.prob_choices = np.array(self.config.prob_choices)
        
        # Statistics
        self.stats = {
            'evaluations': 0,
            'solutions_found': 0,
            'best_violation': float('inf'),
            'prefilter_rejections': 0,
            'cache_hits': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Pre-allocate memory
        self._preallocate_memory()
    
    def _preallocate_memory(self):
        """Pre-allocate memory to reduce GC overhead"""
        self.temp_outcomes = np.empty(6, dtype=np.float64)
        self.temp_probabilities = np.empty(6, dtype=np.float64)
        self.temp_params = np.empty(16, dtype=np.float64)  # CORRECTED: 16 parameters
    
    def generate_batch_params(self, batch_size: int) -> np.ndarray:
        """Generate parameters in batches for 6 outcomes"""
        params_batch = np.empty((batch_size, 16), dtype=np.float64)  # CORRECTED: 16 parameters
        
        # Vectorized parameter generation
        for i in range(batch_size):
            # 11 integer lottery values
            params_batch[i, :11] = np.random.choice(self.lottery_values, 11)
            # 5 probability values
            params_batch[i, 11:] = np.random.choice(self.prob_choices, 5)
        
        return params_batch
    
    def fast_prefilter(self, params_batch: np.ndarray) -> np.ndarray:
        """Fast prefilter for obviously invalid parameters"""
        if not self.config.use_fast_prefilter:
            return params_batch
        
        valid_indices = []
        
        for i in range(len(params_batch)):
            params = params_batch[i]
            
            # Fast basic constraint check
            if NUMBA_AVAILABLE:
                valid, violation = fast_check_basic_constraints(params)
            else:
                valid, violation = self._slow_check_basic_constraints(params)
            
            if valid and violation < self.config.violation_threshold * 2:
                valid_indices.append(i)
            else:
                self.stats['prefilter_rejections'] += 1
        
        return params_batch[valid_indices] if valid_indices else np.empty((0, 16))
    
    def _slow_check_basic_constraints(self, params):
        """Non-JIT version of basic constraint checking"""
        b11, b12, c21, c22, c23, c31, c32, c33, c34, c35, c36, p1, p2, p3, p4, p5 = params
        
        # Probability bounds check
        if not (0 <= p1 <= 1 and 0 <= p2 <= 1 and 0 <= p3 <= 1 and 0 <= p4 <= 1 and 0 <= p5 <= 1):
            return False, 1000.0
        
        # Create lottery structure
        Z, p_array = self.create_lottery_structure(params)
        z1, z2, z3, z4, z5, z6 = Z
        
        # Expected value constraint
        expected_value = np.sum(Z * p_array)
        expected_violation = abs(expected_value)
        
        # Ordering constraints
        ordering_violations = 0.0
        if z1 <= z2:
            ordering_violations += (z2 - z1 + 1)
        if z2 <= 0:
            ordering_violations += (-z2 + 1)
        if z3 <= 0:
            ordering_violations += (-z3 + 1)
        if z3 <= z4:
            ordering_violations += (z4 - z3 + 1)
        if z5 <= 0:
            ordering_violations += (-z5 + 1)
        if z5 <= z6:
            ordering_violations += (z6 - z5 + 1)
        if abs(z2 - z3) < 0.01:
            ordering_violations += 2
        if abs(z4 - z5) < 0.01:
            ordering_violations += 2
        
        total_violation = expected_violation + ordering_violations
        
        return True, total_violation
    
    def create_lottery_structure(self, params):
        """Create lottery structure"""
        if NUMBA_AVAILABLE:
            return fast_create_lottery_structure(params)
        else:
            return self._slow_create_lottery_structure(params)
    
    def _slow_create_lottery_structure(self, params):
        """Non-JIT version of lottery structure creation"""
        b11, b12, c21, c22, c23, c31, c32, c33, c34, c35, c36, p1, p2, p3, p4, p5 = params
        
        # Calculate final outcomes
        z1 = b11 + c21 + c31
        z2 = b11 + c21 + c32
        z3 = b12 + c22 + c33
        z4 = b12 + c22 + c34
        z5 = b12 + c23 + c35
        z6 = b12 + c23 + c36
        
        # Calculate path probabilities
        prob1 = p1 * p2
        prob2 = p1 * (1 - p2)
        prob3 = (1 - p1) * p3 * p4
        prob4 = (1 - p1) * p3 * (1 - p4)
        prob5 = (1 - p1) * (1 - p3) * p5
        prob6 = (1 - p1) * (1 - p3) * (1 - p5)
        
        outcomes = np.array([z1, z2, z3, z4, z5, z6])
        probabilities = np.array([prob1, prob2, prob3, prob4, prob5, prob6])
        
        return outcomes, probabilities
    
    def find_R_where_Y_equals_zero(self, Z, p):
        """Find reference point where Y = 0"""
        def equation(R):
            if NUMBA_AVAILABLE:
                return fast_compute_Y(Z, p, R, self.alpha, self.lambda_, self.gamma)
            else:
                return self._slow_compute_Y(Z, p, R)
        
        # Use better starting points
        starting_points = [0.0, np.mean(Z), np.median(Z)]
        
        for start_R in starting_points:
            try:
                R_solution = fsolve(equation, start_R, xtol=1e-6)[0]
                if abs(equation(R_solution)) < 1e-4:
                    return float(R_solution)
            except:
                continue
        
        return None
    
    def _slow_compute_Y(self, Z, p, R):
        """Non-JIT version of Y computation"""
        # Value function
        diff = Z - R
        v = np.where(diff >= 0,
                     np.power(np.abs(diff) + 1e-12, self.alpha),
                     -self.lambda_ * np.power(np.abs(diff) + 1e-12, self.alpha))
        
        # Probability weighting
        w = np.array([self._probability_weighting(prob) for prob in p])
        
        return float(np.sum(v * w))
    
    def _probability_weighting(self, p):
        """Probability weighting function"""
        p = np.clip(p, 1e-10, 1 - 1e-10)
        numerator = p ** self.gamma
        denominator = (p ** self.gamma + (1 - p) ** self.gamma) ** (1 / self.gamma)
        return numerator / denominator
    
    def find_monotonic_interval(self, Z, p):
        """Find monotonic interval"""
        try:
            R_zero = self.find_R_where_Y_equals_zero(Z, p)
            if R_zero is None:
                return None, None
            
            larger_outcomes = Z[Z > R_zero]
            
            if len(larger_outcomes) == 0:
                next_reference = float(np.max(Z))
            else:
                differences = larger_outcomes - R_zero
                min_diff_idx = np.argmin(differences)
                next_reference = float(larger_outcomes[min_diff_idx])
            
            if next_reference <= R_zero:
                sorted_outcomes = np.sort(Z)
                if len(sorted_outcomes) >= 2:
                    next_reference = float(sorted_outcomes[-2])
                else:
                    next_reference = R_zero + 1.0
            
            return R_zero, next_reference
        except:
            return None, None
    
    def check_full_constraints(self, params):
        """Full constraint checking for 6 outcomes - consistent with original version"""
        try:
            self.stats['evaluations'] += 1
            
            b11, b12, c21, c22, c23, c31, c32, c33, c34, c35, c36, p1, p2, p3, p4, p5 = params
            Z, p = self.create_lottery_structure(params)
            z1, z2, z3, z4, z5, z6 = Z
            
            violations = {}
            total_violations = 0
            
            # Constraint 1: Zero initial expectation
            expected_value = np.sum(Z * p)
            violations['expected_value'] = (abs(expected_value))**2
            total_violations += violations['expected_value']
            
            # Constraint 5: Ordering constraints for 6 outcomes
            ordering_violations = 0
            if not (z1 > z2):
                ordering_violations += (z2 - z1 + 1)
            if not (0 < z2):
                ordering_violations += (-z2 + 1)
            if not (z3 > 0):
                ordering_violations += (-z3 + 1)
            if not (z3 > z4):
                ordering_violations += (z4 - z3 + 1)
            if not (z5 > 0):
                ordering_violations += (-z5 + 1)
            if not (z5 > z6):
                ordering_violations += (z6 - z5 + 1)
            if abs(z2 - z3) < 0.01:
                ordering_violations += 2
            if abs(z4 - z5) < 0.01:
                ordering_violations += 2
            
            violations['ordering'] = ordering_violations
            total_violations += ordering_violations
            
            # Calculate expected values for sub-lotteries
            E_L1 = z1 * p2 + z2 * (1 - p2)  # Upper lottery expected value
            E_L2 = z3 * p4 + z4 * (1 - p4)  # Lower first sub-lottery expected value
            E_L3 = z5 * p5 + z6 * (1 - p5)  # Lower second sub-lottery expected value
            E_L4 = p3 * E_L2 + (1 - p3) * E_L3  # Lower lottery expected value
            
            # Constraint 2: Monotonic interval for main lottery L
            IL_lower, IL_upper = self.find_monotonic_interval(Z, p)
            if IL_lower is None or IL_upper is None or IL_lower >= IL_upper:
                violations['empty_interval'] = 1000
                total_violations += 1000
                violations['total'] = total_violations
                return violations, False
            
            # Check 0 in IL
            values_to_check_L = [0]
            interval_violations_L = 0
            for value in values_to_check_L:
                if value < IL_lower:
                    interval_violations_L += (IL_lower - value)
                elif value > IL_upper:
                    interval_violations_L += (value - IL_upper)
            
            violations['interval_L'] = interval_violations_L
            total_violations += interval_violations_L
            
            # Constraint 3: Monotonic interval for L1 (upper lottery)
            Z_L1 = np.array([z1, z2])
            p_L1 = np.array([p2, 1 - p2])
            IL1_lower, IL1_upper = self.find_monotonic_interval(Z_L1, p_L1)
            
            if IL1_lower is None or IL1_upper is None or IL1_lower >= IL1_upper:
                violations['empty_interval_L1'] = 100
                total_violations += 100
            else:
                values_to_check_L1 = [b11, b11 + c21, E_L1, expected_value]
                interval_violations_L1 = 0
                for value in values_to_check_L1:
                    if value < IL1_lower:
                        interval_violations_L1 += (IL1_lower - value)
                    elif value > IL1_upper:
                        interval_violations_L1 += (value - IL1_upper)
                
                violations['interval_L1'] = interval_violations_L1
                total_violations += interval_violations_L1
            
            # Constraint 4: Monotonic interval for L2 (lower first sub-lottery)
            Z_L2 = np.array([z3, z4])
            p_L2 = np.array([p4, 1 - p4])
            IL2_lower, IL2_upper = self.find_monotonic_interval(Z_L2, p_L2)
            
            if IL2_lower is None or IL2_upper is None or IL2_lower >= IL2_upper:
                violations['empty_interval_L2'] = 100
                total_violations += 100
            else:
                values_to_check_L2 = [b12, b12 + c22, E_L2, E_L4]
                interval_violations_L2 = 0
                for value in values_to_check_L2:
                    if value < IL2_lower:
                        interval_violations_L2 += (IL2_lower - value)
                    elif value > IL2_upper:
                        interval_violations_L2 += (value - IL2_upper)
                
                violations['interval_L2'] = interval_violations_L2
                total_violations += interval_violations_L2
            
            # Constraint 5: Monotonic interval for L3 (lower second sub-lottery)
            Z_L3 = np.array([z5, z6])
            p_L3 = np.array([p5, 1 - p5])
            IL3_lower, IL3_upper = self.find_monotonic_interval(Z_L3, p_L3)
            
            if IL3_lower is None or IL3_upper is None or IL3_lower >= IL3_upper:
                violations['empty_interval_L3'] = 100
                total_violations += 100
            else:
                values_to_check_L3 = [b12, b12 + c23, E_L3, E_L4]
                interval_violations_L3 = 0
                for value in values_to_check_L3:
                    if value < IL3_lower:
                        interval_violations_L3 += (IL3_lower - value)
                    elif value > IL3_upper:
                        interval_violations_L3 += (value - IL3_upper)
                
                violations['interval_L3'] = interval_violations_L3
                total_violations += interval_violations_L3
            
            # Constraint 6: Monotonic interval for L4 (lower compound lottery)
            Z_L4 = np.array([z3, z4, z5, z6])
            p_L4 = np.array([p3 * p4, p3 * (1-p4), (1-p3) * p5, (1-p3) * (1-p5)])
            IL4_lower, IL4_upper = self.find_monotonic_interval(Z_L4, p_L4)
            
            if IL4_lower is None or IL4_upper is None or IL4_lower >= IL4_upper:
                violations['empty_interval_L4'] = 100
                total_violations += 100
            else:
                values_to_check_L4 = [b12, E_L4, expected_value]
                interval_violations_L4 = 0
                for value in values_to_check_L4:
                    if value < IL4_lower:
                        interval_violations_L4 += (IL4_lower - value)
                    elif value > IL4_upper:
                        interval_violations_L4 += (value - IL4_upper)
                
                violations['interval_L4'] = interval_violations_L4
                total_violations += interval_violations_L4
            
            # Probability bounds constraints
            prob_bound_violations = 0
            for prob in [p1, p2, p3, p4, p5]:
                if prob < 0:
                    prob_bound_violations += (-prob)
                if prob > 1:
                    prob_bound_violations += (prob - 1)
            
            violations['prob_bounds'] = prob_bound_violations
            total_violations += prob_bound_violations
            
            violations['total'] = total_violations
            
            # Update best violation
            if total_violations < self.stats['best_violation']:
                self.stats['best_violation'] = total_violations
            
            return violations, True
            
        except Exception as e:
            return {'total': 10000, 'error': str(e)}, False
    
    def batch_optimize(self, batch_size: int = None) -> List[CompleteSolution]:
        """Batch optimization"""
        if batch_size is None:
            batch_size = self.config.batch_size
        
        solutions = []
        attempts_made = 0
        
        self.stats['start_time'] = time.time()
        
        pbar = tqdm(total=self.config.num_attempts, desc="Batch optimization")
        
        while attempts_made < self.config.num_attempts:
            current_batch_size = min(batch_size, self.config.num_attempts - attempts_made)
            
            # Generate batch parameters
            params_batch = self.generate_batch_params(current_batch_size)
            
            # Fast prefilter
            filtered_params = self.fast_prefilter(params_batch)
            
            # Full constraint check
            for params in filtered_params:
                violations, valid = self.check_full_constraints(params)
                
                if valid and violations['total'] < self.config.violation_threshold:
                    solution = CompleteSolution(params, violations['total'])
                    solutions.append(solution)
                    self.stats['solutions_found'] += 1
                    
                    # Early termination check
                    if len(solutions) >= self.config.early_termination_solutions:
                        pbar.close()
                        self.stats['end_time'] = time.time()
                        return solutions
            
            attempts_made += current_batch_size
            pbar.update(current_batch_size)
            
            # Update progress bar
            pbar.set_postfix({
                'Solutions': len(solutions),
                'Best': f'{self.stats["best_violation"]:.3f}',
                'Prefilter': f'{self.stats["prefilter_rejections"]}'
            })
        
        pbar.close()
        self.stats['end_time'] = time.time()
        return solutions
    
    def solve(self, method: str = 'batch') -> List[CompleteSolution]:
        """Main solving method"""
        print(f"Starting lottery parameter optimization for 6 outcomes...")
        print(f"Method: {method}")
        print(f"Numba acceleration: {'Enabled' if NUMBA_AVAILABLE else 'Disabled'}")
        print(f"Prospect theory parameters: alpha={self.alpha}, lambda={self.lambda_}, gamma={self.gamma}")
        print(f"Parameter space: 11 lottery values + 5 probabilities = 16 parameters")
        
        if method == 'batch':
            solutions = self.batch_optimize()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Sort solutions
        solutions.sort(key=lambda s: s.violation)
        
        elapsed_time = (self.stats['end_time'] or time.time()) - self.stats['start_time']
        
        print(f"\nOptimization completed ({elapsed_time:.2f}s)")
        print(f"Statistics:")
        print(f"   Total evaluations: {self.stats['evaluations']:,}")
        print(f"   Solutions found: {len(solutions)}")
        print(f"   Prefilter rejections: {self.stats['prefilter_rejections']:,}")
        print(f"   Evaluation speed: {self.stats['evaluations'] / elapsed_time:.0f} evals/s")
        if solutions:
            print(f"   Best violation value: {solutions[0].violation:.6f}")
        
        return solutions
    
    def calculate_solution_diversity(self, solutions: List[CompleteSolution]) -> Dict[str, float]:
        """Calculate solution diversity metrics"""
        if len(solutions) < 2:
            return {'diversity': 0.0, 'num_solutions': len(solutions)}
        
        # Extract parameter arrays
        params_array = np.array([s.params for s in solutions])
        
        # Calculate pairwise distances
        distances = pdist(params_array, metric='euclidean')
        
        # Calculate diversity metrics
        diversity_metrics = {
            'num_solutions': len(solutions),
            'mean_distance': float(np.mean(distances)),
            'min_distance': float(np.min(distances)),
            'max_distance': float(np.max(distances)),
            'std_distance': float(np.std(distances)),
            'unique_lottery_combinations': len(set(tuple(s.lottery_values) for s in solutions)),
            'unique_prob_combinations': len(set(tuple(s.probabilities) for s in solutions))
        }
        
        return diversity_metrics
    
    def save_solutions_to_file(self, solutions: List[CompleteSolution], filename: str = None) -> str:
        """Save solutions to text file for 6 outcomes"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.config.output_dir, f"lottery_6outcomes_{timestamp}.txt")
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # Write file header
                f.write("=" * 80 + "\n")
                f.write("High-Performance Discrete Lottery Optimizer - 6 Outcomes Results\n")
                f.write("=" * 80 + "\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Solutions found: {len(solutions)}\n")
                f.write(f"Prospect theory parameters: alpha={self.alpha}, lambda={self.lambda_}, gamma={self.gamma}\n")
                f.write(f"Search configuration:\n")
                f.write(f"  - Lottery range: [{self.config.lottery_min}, {self.config.lottery_max}]\n")
                f.write(f"  - Probability choices: {self.config.prob_choices}\n")
                f.write(f"  - Violation threshold: {self.config.violation_threshold}\n")
                f.write(f"  - Total evaluations: {self.stats['evaluations']:,}\n")
                f.write(f"  - Parameter space: 11 lottery values + 5 probabilities = 16 parameters\n")
                f.write(f"  - Outcomes: 6 (z1, z2, z3, z4, z5, z6)\n")
                f.write(f"  - Numba acceleration: {'Enabled' if NUMBA_AVAILABLE else 'Disabled'}\n")
                f.write("\n")
                
                if not solutions:
                    f.write("No solutions found!\n")
                    f.write("Suggestions:\n")
                    f.write("  - Increase num_attempts\n")
                    f.write("  - Relax violation_threshold\n")
                    f.write("  - Adjust parameter ranges\n")
                    return filename
                
                # Solution details
                for i, solution in enumerate(solutions):
                    f.write(f"\n{'='*60}\n")
                    f.write(f"Solution {i+1}\n")
                    f.write(f"{'='*60}\n")
                    f.write(f"Objective function value: {solution.violation:.6f}\n")
                    f.write(f"Timestamp: {datetime.fromtimestamp(solution.timestamp).strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("\n")
                    
                    # Lottery structure for 6 outcomes
                    f.write("Lottery structure (6 outcomes):\n")
                    f.write(f"  Stage 1: b11={solution.b11:3d}, b12={solution.b12:3d}\n")
                    f.write(f"  Stage 2: c21={solution.c21:3d}, c22={solution.c22:3d}, c23={solution.c23:3d}\n")
                    f.write(f"  Stage 3: c31={solution.c31:3d}, c32={solution.c32:3d}, c33={solution.c33:3d}\n")
                    f.write(f"           c34={solution.c34:3d}, c35={solution.c35:3d}, c36={solution.c36:3d}\n")
                    f.write(f"  Probabilities: p1={solution.p1:.3f}, p2={solution.p2:.3f}, p3={solution.p3:.3f}\n")
                    f.write(f"                 p4={solution.p4:.3f}, p5={solution.p5:.3f}\n")
                    f.write("\n")
                    
                    # Calculate final outcomes
                    Z, p = self.create_lottery_structure(solution.params)
                    z1, z2, z3, z4, z5, z6 = Z
                    expected_value = np.sum(Z * p)
                    
                    f.write("Final outcomes (6 outcomes):\n")
                    f.write(f"  z1 = 0+{solution.b11}+{solution.c21}+{solution.c31} = {int(z1)} (prob={p[0]:.3f})\n")
                    f.write(f"  z2 = 0+{solution.b11}+{solution.c21}+{solution.c32} = {int(z2)} (prob={p[1]:.3f})\n")
                    f.write(f"  z3 = 0+{solution.b12}+{solution.c22}+{solution.c33} = {int(z3)} (prob={p[2]:.3f})\n")
                    f.write(f"  z4 = 0+{solution.b12}+{solution.c22}+{solution.c34} = {int(z4)} (prob={p[3]:.3f})\n")
                    f.write(f"  z5 = 0+{solution.b12}+{solution.c23}+{solution.c35} = {int(z5)} (prob={p[4]:.3f})\n")
                    f.write(f"  z6 = 0+{solution.b12}+{solution.c23}+{solution.c36} = {int(z6)} (prob={p[5]:.3f})\n")
                    f.write(f"  Expected value: {expected_value:.6f}\n")
                    f.write(f"  Probability sum: {np.sum(p):.6f} (should be 1.0)\n")
                    f.write("\n")
                    
                    # Constraint verification
                    violations, valid = self.check_full_constraints(solution.params)
                    f.write("Constraint verification:\n")
                    f.write(f"  1. Expected value constraint: {expected_value:.6f} ≈ 0 {'✓' if abs(expected_value) < 0.5 else '✗'}\n")
                    f.write(f"  2. Ordering constraints: {'✓' if violations.get('ordering', 0) < 1 else '✗'}\n")
                    f.write(f"     z1 > 0 > z2 : {z1} > 0 > {z2}  = {z1 > 0 > z2 }\n")
                    f.write(f"     z3  > 0  > z4: {z3} > 0 > {z4}  = {z3   > 0 > z4}\n")
                    f.write(f"     z5 > 0 > z6 : {z5} > 0 > {z6}  = {z5  > 0 > z6 }\n")
                    f.write(f"  3. Probability constraint: all probabilities ∈ [0,1] {'✓' if all(0 <= p <= 1 for p in solution.probabilities) else '✗'}\n")
                    
                    # Calculate sub-lottery expected values
                    E_L1 = z1 * solution.p2 + z2 * (1 - solution.p2)
                    E_L2 = z3 * solution.p4 + z4 * (1 - solution.p4)
                    E_L3 = z5 * solution.p5 + z6 * (1 - solution.p5)
                    E_L4 = solution.p3 * E_L2 + (1 - solution.p3) * E_L3
                    
                    f.write(f"\nSub-lottery expected values:\n")
                    f.write(f"  E(L1) = {E_L1:.3f} (upper lottery)\n")
                    f.write(f"  E(L2) = {E_L2:.3f} (lower first sub-lottery)\n")
                    f.write(f"  E(L3) = {E_L3:.3f} (lower second sub-lottery)\n")
                    f.write(f"  E(L4) = {E_L4:.3f} (lower compound lottery)\n")
                    
                    # Interval information
                    IL_lower, IL_upper = self.find_monotonic_interval(Z, p)
                    if IL_lower is not None and IL_upper is not None:
                        f.write(f"\n  4. Main interval IL = [{IL_lower:.3f}, {IL_upper:.3f}]\n")
                        f.write(f"     0 ∈ IL: {IL_lower <= 0 <= IL_upper}\n")
                    
                    # Sub-lottery intervals
                    Z_L1 = np.array([z1, z2])
                    p_L1 = np.array([solution.p2, 1 - solution.p2])
                    IL1_lower, IL1_upper = self.find_monotonic_interval(Z_L1, p_L1)
                    if IL1_lower is not None and IL1_upper is not None:
                        f.write(f"  5. L1 interval IL1 = [{IL1_lower:.3f}, {IL1_upper:.3f}]\n")
                        values_L1 = [solution.b11, solution.b11 + solution.c21, E_L1, expected_value]
                        value_names_L1 = ['b11', 'b11+c21', 'E(L1)', 'E(L)']
                        for j, value in enumerate(values_L1):
                            in_interval = IL1_lower <= value <= IL1_upper
                            f.write(f"     {'✓' if in_interval else '✗'} {value_names_L1[j]} = {value:.3f} {'∈' if in_interval else '∉'} IL1\n")
                    
                    Z_L2 = np.array([z3, z4])
                    p_L2 = np.array([solution.p4, 1 - solution.p4])
                    IL2_lower, IL2_upper = self.find_monotonic_interval(Z_L2, p_L2)
                    if IL2_lower is not None and IL2_upper is not None:
                        f.write(f"  6. L2 interval IL2 = [{IL2_lower:.3f}, {IL2_upper:.3f}]\n")
                        values_L2 = [solution.b12, solution.b12 + solution.c22, E_L2, E_L4]
                        value_names_L2 = ['b12', 'b12+c22', 'E(L2)', 'E(L4)']
                        for j, value in enumerate(values_L2):
                            in_interval = IL2_lower <= value <= IL2_upper
                            f.write(f"     {'✓' if in_interval else '✗'} {value_names_L2[j]} = {value:.3f} {'∈' if in_interval else '∉'} IL2\n")
                    
                    Z_L3 = np.array([z5, z6])
                    p_L3 = np.array([solution.p5, 1 - solution.p5])
                    IL3_lower, IL3_upper = self.find_monotonic_interval(Z_L3, p_L3)
                    if IL3_lower is not None and IL3_upper is not None:
                        f.write(f"  7. L3 interval IL3 = [{IL3_lower:.3f}, {IL3_upper:.3f}]\n")
                        values_L3 = [solution.b12, solution.b12 + solution.c23, E_L3, E_L4]
                        value_names_L3 = ['b12', 'b12+c23', 'E(L3)', 'E(L4)']
                        for j, value in enumerate(values_L3):
                            in_interval = IL3_lower <= value <= IL3_upper
                            f.write(f"     {'✓' if in_interval else '✗'} {value_names_L3[j]} = {value:.3f} {'∈' if in_interval else '∉'} IL3\n")
                
                # Summary table
                f.write(f"\n{'='*80}\n")
                f.write("Solution Summary Table (6 Outcomes)\n")
                f.write(f"{'='*80}\n")
                
                # Create summary data
                summary_data = []
                for i, sol in enumerate(solutions):
                    Z, p = self.create_lottery_structure(sol.params)
                    expected_value = np.sum(Z * p)
                    summary_data.append({
                        'Sol': i+1,
                        'b11': sol.b11, 'b12': sol.b12,
                        'c21': sol.c21, 'c22': sol.c22, 'c23': sol.c23,
                        'c31': sol.c31, 'c32': sol.c32, 'c33': sol.c33,
                        'c34': sol.c34, 'c35': sol.c35, 'c36': sol.c36,
                        'p1': sol.p1, 'p2': sol.p2, 'p3': sol.p3, 'p4': sol.p4, 'p5': sol.p5,
                        'z1': int(Z[0]), 'z2': int(Z[1]), 'z3': int(Z[2]),
                        'z4': int(Z[3]), 'z5': int(Z[4]), 'z6': int(Z[5]),
                        'E[Z]': expected_value,
                        'Violation': sol.violation
                    })
                
                df = pd.DataFrame(summary_data)
                f.write(df.to_string(index=False, float_format='%.3f'))
                f.write("\n")
                
                # Diversity metrics
                diversity = self.calculate_solution_diversity(solutions)
                f.write(f"\n{'='*80}\n")
                f.write("Solution Diversity Metrics\n")
                f.write(f"{'='*80}\n")
                f.write(f"Total solutions: {diversity['num_solutions']}\n")
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
                f.write(f"Success rate: {self.stats['solutions_found'] / self.stats['evaluations'] * 100:.3f}%\n")
                f.write(f"Prefilter rejections: {self.stats['prefilter_rejections']:,}\n")
                f.write(f"Evaluation speed: {self.stats['evaluations'] / elapsed_time:.0f} evals/s\n")
                
                if solutions:
                    violations = [sol.violation for sol in solutions]
                    f.write(f"Best violation value: {min(violations):.6f}\n")
                    f.write(f"Average violation value: {np.mean(violations):.6f}\n")
                    f.write(f"Worst violation value: {max(violations):.6f}\n")
                
                # Search space coverage
                total_space = len(self.lottery_values) ** 11 * len(self.prob_choices) ** 5
                searched_fraction = self.stats['evaluations'] / total_space * 100
                f.write(f"Search space coverage: {searched_fraction:.2e}%\n")
                
                # Lottery structure explanation
                f.write(f"\n{'='*80}\n")
                f.write("Lottery Structure Explanation (6 Outcomes)\n")
                f.write(f"{'='*80}\n")
                f.write("Decision tree structure:\n")
                f.write("  Initial choice (prob p1):\n")
                f.write("    ├─ Upper branch (prob p1): Gets b11\n")
                f.write("    │   └─ Sub-choice (prob p2):\n")
                f.write("    │       ├─ Get c21+c31 → Final: z1 = b11+c21+c31\n")
                f.write("    │       └─ Get c21+c32 → Final: z2 = b11+c21+c32\n")
                f.write("    └─ Lower branch (prob 1-p1): Gets b12\n")
                f.write("        └─ Sub-choice (prob p3):\n")
                f.write("            ├─ First sub-lottery (prob p3): Gets c22\n")
                f.write("            │   └─ Final choice (prob p4):\n")
                f.write("            │       ├─ Get c33 → Final: z3 = b12+c22+c33\n")
                f.write("            │       └─ Get c34 → Final: z4 = b12+c22+c34\n")
                f.write("            └─ Second sub-lottery (prob 1-p3): Gets c23\n")
                f.write("                └─ Final choice (prob p5):\n")
                f.write("                    ├─ Get c35 → Final: z5 = b12+c23+c35\n")
                f.write("                    └─ Get c36 → Final: z6 = b12+c23+c36\n")
                
                f.write(f"\nFile saved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n")
            
            return filename
            
        except Exception as e:
            print(f"Error saving file: {e}")
            return None
    
    def save_progress(self, solutions: List[CompleteSolution], filename: str = None):
        """Save progress to pickle file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(self.config.output_dir, f"lottery_6outcomes_progress_{timestamp}.pkl")
        
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
    
    def load_progress(self, filename: str):
        """Load progress from pickle file"""
        try:
            with open(filename, 'rb') as f:
                progress_data = pickle.load(f)
            
            # Restore solutions
            solutions = []
            for sol_dict in progress_data['solutions']:
                solution = CompleteSolution(
                    np.array(sol_dict['params']),
                    sol_dict['violation'],
                    sol_dict['timestamp']
                )
                solutions.append(solution)
            
            # Restore statistics
            self.stats.update(progress_data['stats'])
            
            print(f"Loaded {len(solutions)} solutions")
            return solutions
            
        except Exception as e:
            print(f"Error loading progress: {e}")
            return []


def main():
    """Main function for 6-outcome lottery optimization"""
    print("High-Performance Discrete Lottery Optimizer (6 Outcomes)")
    print("=" * 60)
    
    # Configuration for 6 outcomes
    config = OptimizedConfig(
        alpha=0.88,
        lambda_=2.25,
        gamma=0.61,
        num_attempts=1000000000,
        violation_threshold=20.0,
        prob_choices=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        batch_size=5000,
        early_termination_solutions=50,
        use_fast_prefilter=True,
        output_dir="lottery_results_6outcomes"
    )
    
    # Create optimizer
    optimizer = OptimizedLotteryOptimizer(config)
    
    # Run optimization
    print("Starting optimization for 6-outcome lottery...")
    print(f"Parameter space: {len(optimizer.lottery_values)}^11 × {len(optimizer.prob_choices)}^5")
    print(f"Total combinations: ~{len(optimizer.lottery_values)**11 * len(optimizer.prob_choices)**5:.2e}")
    
    solutions = optimizer.solve(method='batch')
    
    # Save results to file
    if solutions:
        print(f"\nFound {len(solutions)} solutions for 6-outcome lottery")
        
        # Save detailed results to txt file
        txt_filename = optimizer.save_solutions_to_file(solutions)
        if txt_filename:
            print(f"Detailed results saved to: {txt_filename}")
        
        # Save progress (optional)
        save_progress = input("\nSave progress file? (y/n): ").strip().lower()
        if save_progress == 'y':
            pkl_filename = optimizer.save_progress(solutions)
            if pkl_filename:
                print(f"Progress file saved to: {pkl_filename}")
        
        # Display first few solutions
        print(f"\nTop {min(3, len(solutions))} best solutions:")
        for i, sol in enumerate(solutions[:3]):
            Z, p = optimizer.create_lottery_structure(sol.params)
            print(f"  {i+1}. Violation: {sol.violation:.6f}")
            print(f"      Outcomes: z={[int(z) for z in Z]}")
            print(f"      Probabilities: p={[f'{prob:.3f}' for prob in p]}")
            print(f"      Expected value: {np.sum(Z * p):.6f}")
        
        print(f"\nCompleted! See generated txt file for detailed results.")
        
    else:
        print("\nNo solutions found for 6-outcome lottery")
        print("Suggestions:")
        print("  - Increase num_attempts")
        print("  - Relax violation_threshold")
        print("  - Adjust parameter ranges")
        
        # Still create a results file to record the attempt
        txt_filename = optimizer.save_solutions_to_file([])
        if txt_filename:
            print(f"Search record saved to: {txt_filename}")
    
    return solutions


if __name__ == "__main__":
    solutions = main()