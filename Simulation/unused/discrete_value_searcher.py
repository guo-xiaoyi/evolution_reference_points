# -*- coding: utf-8 -*-
"""
Pure Discrete Lottery Optimizer - Enhanced Version

Key features:
1. Direct discrete parameter generation (no rounding)
2. Multiple discrete optimization methods
3. Input validation and error handling
4. Performance optimizations with caching
5. Progress saving and resumption
6. Solution diversity metrics
7. Adaptive constraint relaxation

@author: XGuo xiaoyi.guo@unisg.ch (original)
@enhanced_by: Claude (discrete optimization with improvements)
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
from functools import partial, lru_cache
import pickle
from scipy.spatial.distance import pdist

warnings.filterwarnings('ignore')


@dataclass
class DiscreteOptimizationConfig:
    """Configuration for discrete optimization with validation"""
    alpha: float = 0.88
    lambda_: float = 2.25
    gamma: float = 0.61

    # Discrete parameter ranges
    lottery_min: int = -50
    lottery_max: int = 50
    prob_choices: List[float] = None

    # Optimization settings
    num_attempts: int = 10000
    violation_threshold: float = 5.0
    num_cores: Optional[int] = None

    # Advanced settings
    use_adaptive_search: bool = True
    use_local_improvement: bool = True
    tournament_size: int = 3

    def __post_init__(self):
        if self.prob_choices is None:
            self.prob_choices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        # Validation
        if self.lottery_min >= self.lottery_max:
            raise ValueError(f"lottery_min ({self.lottery_min}) must be less than lottery_max ({self.lottery_max})")

        if not all(0 <= p <= 1 for p in self.prob_choices):
            raise ValueError("All probability choices must be between 0 and 1")

        if self.alpha <= 0 or self.alpha > 1:
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha}")

        if self.lambda_ <= 0:
            raise ValueError(f"lambda must be positive, got {self.lambda_}")

        if self.gamma <= 0 or self.gamma > 1:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma}")


class DiscreteSolution:
    """Class to represent a discrete solution"""

    def __init__(self, params: List[float], violation: float, timestamp: float = None):
        self.x = np.array(params, dtype=float)
        self.fun = float(violation)
        self.timestamp = timestamp or time.time()

        # Extract components for easy access
        self.lottery_values = [int(x) for x in self.x[:8]]
        self.probabilities = list(self.x[8:])

    def __str__(self):
        return f"Solution(violation={self.fun:.6f}, lottery={self.lottery_values}, probs={self.probabilities})"

    def to_dict(self):
        """Convert solution to dictionary for serialization"""
        return {
            'params': self.x.tolist(),
            'violation': self.fun,
            'timestamp': self.timestamp,
            'lottery_values': self.lottery_values,
            'probabilities': self.probabilities
        }

    @classmethod
    def from_dict(cls, data: Dict):
        """Create solution from dictionary"""
        return cls(data['params'], data['violation'], data.get('timestamp'))


def unified_worker_function(args):
    """Unified worker function for all parallel operations"""
    try:
        task_type, *task_args = args

        if task_type in ["evaluate", "explore", "exploit"]:
            params, config_dict = task_args
            config = DiscreteOptimizationConfig(**config_dict)
            optimizer = DiscreteLotteryOptimizer(config)
            violation = optimizer.objective_function(params)

            threshold = config.violation_threshold
            if task_type == "explore":
                threshold *= 2  # More lenient for exploration

            return (params, violation) if violation < threshold else None

        elif task_type == "random_search":
            seed, violation_threshold, config_dict = task_args
            np.random.seed(seed)
            random.seed(seed)
            config = DiscreteOptimizationConfig(**config_dict)
            optimizer = DiscreteLotteryOptimizer(config)
            params = optimizer.generate_random_discrete_params()
            violation = optimizer.objective_function(params)

            if violation < violation_threshold:
                return {
                    'params': params,
                    'violation': violation,
                    'seed': seed,
                    'timestamp': time.time()
                }
            return None

    except Exception as e:
        return {'error': str(e), 'task_type': task_type}


class DiscreteLotteryOptimizer:
    """Pure discrete lottery optimizer with enhancements"""

    def __init__(self, config: DiscreteOptimizationConfig = None):
        if config is None:
            config = DiscreteOptimizationConfig()

        self.config = config
        self.alpha = config.alpha
        self.lambda_ = config.lambda_
        self.gamma = config.gamma

        # Discrete parameter spaces
        self.lottery_values = list(range(config.lottery_min, config.lottery_max + 1))
        self.prob_choices = config.prob_choices

        # Statistics tracking
        self.stats = {
            'evaluations': 0,
            'solutions_found': 0,
            'best_violation': float('inf'),
            'start_time': None,
            'cache_hits': 0,
            'cache_misses': 0
        }

        # Clear cache for new instance
        self._cached_probability_weighting.cache_clear()

    def generate_random_discrete_params(self) -> List[float]:
        """Generate random discrete parameters - no continuous values!"""

        # 8 integer lottery values
        lottery_params = [float(random.choice(self.lottery_values)) for _ in range(8)]

        # 3 discrete probability values
        prob_params = [random.choice(self.prob_choices) for _ in range(3)]

        return lottery_params + prob_params

    def generate_systematic_params(self, strategy: str = 'latin_hypercube') -> List[List[float]]:
        """Generate systematic parameter combinations"""

        if strategy == 'latin_hypercube':
            return self._latin_hypercube_discrete()
        elif strategy == 'grid_sample':
            return self._grid_sample_discrete()
        else:
            # Fallback to random
            return [self.generate_random_discrete_params() for _ in range(100)]

    def _latin_hypercube_discrete(self, n_samples: int = 500) -> List[List[float]]:
        """Latin hypercube sampling in discrete space"""
        samples = []

        for _ in range(n_samples):
            # Stratified sampling for better coverage
            lottery_params = []
            for i in range(8):
                # Divide lottery range into n_samples strata
                stratum_size = len(self.lottery_values) // n_samples
                if stratum_size == 0:
                    stratum_size = 1
                start_idx = (i * stratum_size) % len(self.lottery_values)
                end_idx = min(start_idx + stratum_size, len(self.lottery_values))
                lottery_params.append(float(random.choice(self.lottery_values[start_idx:end_idx])))

            # Random probability selection (already discrete)
            prob_params = [random.choice(self.prob_choices) for _ in range(3)]

            samples.append(lottery_params + prob_params)

        return samples

    def _grid_sample_discrete(self, n_per_dim: int = 5) -> List[List[float]]:
        """Grid sampling in discrete space"""
        # Sample n_per_dim values from each dimension
        lottery_samples = []
        for _ in range(8):
            step = max(1, len(self.lottery_values) // n_per_dim)
            samples = self.lottery_values[::step][:n_per_dim]
            lottery_samples.append(samples)

        prob_samples = []
        for _ in range(3):
            step = max(1, len(self.prob_choices) // n_per_dim)
            samples = self.prob_choices[::step][:n_per_dim]
            prob_samples.append(samples)

        # Generate combinations (limited to avoid explosion)
        import itertools
        all_samples = []
        for combo in itertools.product(*lottery_samples, *prob_samples):
            all_samples.append(list(combo))
            if len(all_samples) >= 1000:  # Limit number of samples
                break

        return all_samples

    def create_lottery_structure(self, params: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Create lottery structure - same as original but with type safety"""
        if len(params) != 11:
            raise ValueError(f"Expected 11 parameters, got {len(params)}")

        b11, b12, c21, c22, c31, c32, c33, c34, p1, p2, p3 = params

        # Ensure integer lottery values
        b11, b12, c21, c22, c31, c32, c33, c34 = [int(x) for x in [b11, b12, c21, c22, c31, c32, c33, c34]]

        a = 0  # Starting point

        # Calculate final outcomes
        z1 = a + b11 + c21 + c31
        z2 = a + b11 + c21 + c32
        z3 = a + b12 + c22 + c33
        z4 = a + b12 + c22 + c34

        # Calculate path probabilities
        prob1 = p1 * p2
        prob2 = p1 * (1 - p2)
        prob3 = (1 - p1) * p3
        prob4 = (1 - p1) * (1 - p3)

        outcomes = np.array([z1, z2, z3, z4], dtype=float)
        probabilities = np.array([prob1, prob2, prob3, prob4], dtype=float)

        return outcomes, probabilities

    def value_function(self, Z: np.ndarray, R: float) -> np.ndarray:
        """Prospect theory value function"""
        diff = Z - R
        return np.where(diff >= 0,
                        np.power(np.abs(diff) + 1e-12, self.alpha),
                        -self.lambda_ * np.power(np.abs(diff) + 1e-12, self.alpha))

    @lru_cache(maxsize=1024)
    def _cached_probability_weighting(self, p: float) -> float:
        """Cached probability weighting for single probability"""
        p = np.clip(p, 1e-10, 1 - 1e-10)
        numerator = p ** self.gamma
        denominator = (p ** self.gamma + (1 - p) ** self.gamma) ** (1 / self.gamma)
        self.stats['cache_hits'] += 1
        return numerator / denominator

    def probability_weighting(self, p: np.ndarray) -> np.ndarray:
        """Probability weighting function using cache"""
        if isinstance(p, np.ndarray):
            return np.array([self._cached_probability_weighting(float(prob)) for prob in p])
        else:
            return self._cached_probability_weighting(float(p))

    def compute_Y(self, Z: np.ndarray, p: np.ndarray, R: float) -> float:
        """Compute prospect theory value"""
        try:
            v = self.value_function(Z, R)
            w = self.probability_weighting(p)
            return float(np.sum(v * w))
        except:
            return np.inf

    def find_R_where_Y_equals_zero(self, Z: np.ndarray, p: np.ndarray) -> Optional[float]:
        """Find reference point where Y = 0"""

        def equation(R):
            return self.compute_Y(Z, p, R)

        starting_points = [0.0, np.mean(Z), np.median(Z)]

        for start_R in starting_points:
            try:
                R_solution = fsolve(equation, start_R, xtol=1e-8)[0]
                if abs(equation(R_solution)) < 1e-6:
                    return float(R_solution)
            except:
                continue
        return None

    def find_monotonic_interval(self, Z: np.ndarray, p: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
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

    def check_constraints(self, params: List[float]) -> Tuple[Dict[str, float], bool]:
        """Check all lottery constraints following the EXACT ORIGINAL specification"""
        try:
            self.stats['evaluations'] += 1

            # Extract parameters exactly as in original
            b11, b12, c21, c22, c31, c32, c33, c34, p1, p2, p3 = params
            Z, p = self.create_lottery_structure(params)
            z1, z2, z3, z4 = Z

            violations = {}
            total_violations = 0

            # Constraint 1: Null initial expectation E(L) = 0
            expected_value = np.sum(Z * p)
            violations['expected_value'] = abs(expected_value)
            total_violations += violations['expected_value']

            # Constraint 5: Ordering constraint (using EXACT original logic)
            ordering_violations = 0
            if not (z1 > z2):
                ordering_violations += (z2 - z1 + 1)
            if not (0 < z2):  # ORIGINAL: 0 < z2 (not z2 >= 0)
                ordering_violations += (-z2 + 1)
            if not (z3 > 0):  # ORIGINAL: z3 > 0 (not z3 <= 0)
                ordering_violations += (z3 + 1)
            if not (z3 > z4):
                ordering_violations += (z4 - z3 + 1)
            if abs(z2 - z3) < 0.01:  # z2 ≠ z3
                ordering_violations += 2

            violations['ordering'] = ordering_violations
            total_violations += ordering_violations

            # Calculate E(L1) and E(L2) first (needed for multiple constraints)
            E_L1 = z1 * p2 + z2 * (1 - p2)  # Expected value of upper lottery
            E_L2 = z3 * p3 + z4 * (1 - p3)  # Expected value of lower lottery

            # ===== CONSTRAINT 2: Monotonic interval fulfillment for L =====
            IL_lower, IL_upper = self.find_monotonic_interval(Z, p)
            if IL_lower is None or IL_upper is None or IL_lower >= IL_upper:
                violations['empty_interval'] = 1000
                total_violations += 1000
                # Early return since other intervals depend on this
                violations['total'] = total_violations
                return violations, False

            # Check: 0, E ∈ IL (ORIGINAL: only checks 0, not expected_value)
            values_to_check_L = {
                '0': 0,
            }

            interval_violations_L = 0
            for name, value in values_to_check_L.items():
                if value < IL_lower:
                    interval_violations_L += (IL_lower - value)
                elif value > IL_upper:
                    interval_violations_L += (value - IL_upper)

            violations['interval_L'] = interval_violations_L
            total_violations += interval_violations_L

            # ===== CONSTRAINT 3: Monotonic interval fulfillment for L1 =====
            Z_L1 = np.array([z1, z2])
            p_L1 = np.array([p2, 1 - p2])
            IL1_lower, IL1_upper = self.find_monotonic_interval(Z_L1, p_L1)

            if IL1_lower is None or IL1_upper is None or IL1_lower >= IL1_upper:
                violations['empty_interval_L1'] = 100
                total_violations += 100
            else:
                # Check: b11, b11+c21, E(L1), exp_value ∈ IL1 (EXACT ORIGINAL)
                values_to_check_L1 = {
                    'b11': b11,
                    'b11_c21': b11 + c21,
                    'E_L1': E_L1,
                    'exp_value': expected_value  # ORIGINAL includes this
                }

                interval_violations_L1 = 0
                for name, value in values_to_check_L1.items():
                    if value < IL1_lower:
                        interval_violations_L1 += (IL1_lower - value)
                    elif value > IL1_upper:
                        interval_violations_L1 += (value - IL1_upper)

                violations['interval_L1'] = interval_violations_L1
                total_violations += interval_violations_L1

            # ===== CONSTRAINT 4: Monotonic interval fulfillment for L2 =====
            Z_L2 = np.array([z3, z4])
            p_L2 = np.array([p3, 1 - p3])
            IL2_lower, IL2_upper = self.find_monotonic_interval(Z_L2, p_L2)

            if IL2_lower is None or IL2_upper is None or IL2_lower >= IL2_upper:
                violations['empty_interval_L2'] = 100
                total_violations += 100
            else:
                # Check: b12, b12+c22, E(L2), exp_value ∈ IL2 (EXACT ORIGINAL)
                values_to_check_L2 = {
                    'b12': b12,
                    'b12_c22': b12 + c22,
                    'E_L2': E_L2,
                    'exp_value': expected_value  # ORIGINAL includes this
                }

                interval_violations_L2 = 0
                for name, value in values_to_check_L2.items():
                    if value < IL2_lower:
                        interval_violations_L2 += (IL2_lower - value)
                    elif value > IL2_upper:
                        interval_violations_L2 += (value - IL2_upper)

                violations['interval_L2'] = interval_violations_L2
                total_violations += interval_violations_L2

            # Basic probability bounds [0,1]
            prob_bound_violations = 0
            for i, prob in enumerate([p1, p2, p3]):
                if prob < 0:
                    prob_bound_violations += (-prob)
                if prob > 1:
                    prob_bound_violations += (prob - 1)

            violations['prob_bounds'] = prob_bound_violations
            total_violations += prob_bound_violations

            violations['total'] = total_violations
            return violations, True

        except Exception as e:
            return {'total': 10000, 'error': str(e)}, False

    def check_constraints_batch(self, params_list: List[List[float]]) -> List[Tuple[Dict[str, float], bool]]:
        """Check constraints for multiple parameter sets efficiently"""
        results = []
        for params in params_list:
            results.append(self.check_constraints(params))
        return results

    def objective_function(self, params: List[float]) -> float:
        """Objective function following the ORIGINAL constraint specification"""
        violations, valid = self.check_constraints(params)

        if not valid:
            return 10000.0  # Large penalty for invalid solutions

        # Weighted penalty structure - matching your original
        total_violation = 0.0

        # High penalty for ordering violations (most important)
        total_violation += np.abs(violations.get('ordering', 0))

        # Medium penalty for interval violations
        total_violation += np.abs(violations.get('interval_L', 0))
        total_violation += np.abs(violations.get('interval_L1', 0))
        total_violation += np.abs(violations.get('interval_L2', 0))

        # Expected value should be exactly zero
        total_violation += np.abs(violations.get('expected_value', 0))

        # Probability constraints
        total_violation += np.abs(violations.get('prob_bounds', 0))

        # Penalties for empty intervals
        total_violation += np.abs(violations.get('empty_interval', 0))
        total_violation += np.abs(violations.get('empty_interval_L1', 0))
        total_violation += np.abs(violations.get('empty_interval_L2', 0))

        # Update statistics
        if total_violation < self.stats['best_violation']:
            self.stats['best_violation'] = total_violation

        return float(total_violation)

    def local_improvement(self, solution: DiscreteSolution, max_neighbors: int = 50) -> DiscreteSolution:
        """Local improvement through discrete neighborhood search"""
        best_solution = solution

        for _ in range(max_neighbors):
            # Generate neighbor by changing one parameter
            neighbor_params = solution.x.copy()

            # Randomly select parameter to modify
            param_idx = random.randint(0, 10)

            if param_idx < 8:  # Lottery value
                # Change to random nearby lottery value
                current_val = int(neighbor_params[param_idx])
                nearby_values = [v for v in self.lottery_values
                                 if abs(v - current_val) <= 5 and v != current_val]
                if nearby_values:
                    neighbor_params[param_idx] = float(random.choice(nearby_values))
            else:  # Probability
                # Change to different probability choice
                current_prob = neighbor_params[param_idx]
                other_probs = [p for p in self.prob_choices if p != current_prob]
                neighbor_params[param_idx] = random.choice(other_probs)

            # Evaluate neighbor
            neighbor_violation = self.objective_function(neighbor_params.tolist())

            if neighbor_violation < best_solution.fun:
                best_solution = DiscreteSolution(neighbor_params.tolist(), neighbor_violation)

        return best_solution

    def discrete_random_search(self) -> List[DiscreteSolution]:
        """Pure discrete random search - main optimization method"""
        solutions = []

        print(f"🎯 Starting discrete random search...")
        print(f"Parameters: α={self.alpha}, λ={self.lambda_}, γ={self.gamma}")
        print(f"Search space: lottery values ∈ [{self.config.lottery_min}, {self.config.lottery_max}]")
        print(f"              probabilities ∈ {self.prob_choices}")

        self.stats['start_time'] = time.time()

        with tqdm(range(self.config.num_attempts), desc="Discrete Search") as pbar:
            for attempt in pbar:
                # Generate random discrete parameters
                params = self.generate_random_discrete_params()

                # Evaluate
                violation = self.objective_function(params)

                if violation < self.config.violation_threshold:
                    solution = DiscreteSolution(params, violation)

                    # Check uniqueness
                    is_unique = True
                    for existing in solutions:
                        if np.allclose(solution.x, existing.x, atol=1e-6):
                            is_unique = False
                            break

                    if is_unique:
                        # Apply local improvement if enabled
                        if self.config.use_local_improvement and violation > 0.1:
                            solution = self.local_improvement(solution)

                        solutions.append(solution)
                        self.stats['solutions_found'] += 1

                # Update progress
                if attempt % 100 == 0:
                    elapsed = time.time() - self.stats['start_time']
                    rate = self.stats['evaluations'] / elapsed if elapsed > 0 else 0

                    pbar.set_postfix({
                        'Solutions': len(solutions),
                        'Best': f'{self.stats["best_violation"]:.3f}',
                        'Rate': f'{rate:.0f}/s',
                        'Unique': len(set(str(s.lottery_values + s.probabilities) for s in solutions))
                    })

        return solutions

    def discrete_parallel_search(self) -> List[DiscreteSolution]:
        """Parallel discrete search"""
        num_cores = self.config.num_cores or min(mp.cpu_count(), 8)

        # Prepare arguments
        config_dict = {
            'alpha': self.alpha,
            'lambda_': self.lambda_,
            'gamma': self.gamma,
            'lottery_min': self.config.lottery_min,
            'lottery_max': self.config.lottery_max,
            'prob_choices': self.config.prob_choices,
            'violation_threshold': self.config.violation_threshold
        }

        args_list = [
            ("random_search", seed, self.config.violation_threshold, config_dict)
            for seed in range(self.config.num_attempts)
        ]

        print(f"🚀 Running parallel discrete search with {num_cores} cores...")

        solutions = []
        self.stats['start_time'] = time.time()

        # Process in batches
        batch_size = 500

        with tqdm(total=self.config.num_attempts, desc="Parallel Discrete") as pbar:
            for i in range(0, len(args_list), batch_size):
                batch_args = args_list[i:i + batch_size]

                with ProcessPoolExecutor(max_workers=num_cores) as executor:
                    batch_results = list(executor.map(unified_worker_function, batch_args))

                # Process results
                for result in batch_results:
                    if result is not None and 'params' in result:
                        solution = DiscreteSolution(
                            result['params'],
                            result['violation'],
                            result['timestamp']
                        )
                        solutions.append(solution)
                        self.stats['solutions_found'] += 1

                pbar.update(len(batch_args))
                pbar.set_postfix({
                    'Solutions': len(solutions),
                    'Rate': f'{len(batch_args) / (time.time() - self.stats["start_time"] + 1e-6):.0f}/s'
                })

        return solutions

    def adaptive_discrete_search(self) -> List[DiscreteSolution]:
        """Parallel adaptive search that learns from good solutions"""
        solutions = []
        phase1_attempts = self.config.num_attempts // 3
        phase2_attempts = self.config.num_attempts - phase1_attempts

        print("🔍 Phase 1: Random exploration (parallel)")

        # Prepare config dict
        config_dict = self.config.__dict__.copy()

        # Phase 1: Generate random exploration parameters
        param_list_phase1 = []
        for _ in range(phase1_attempts):
            params = self.generate_random_discrete_params()
            param_list_phase1.append(("explore", params, config_dict))

        # Run parallel evaluation
        with ProcessPoolExecutor(max_workers=self.config.num_cores or mp.cpu_count()) as executor:
            results_phase1 = list(tqdm(
                executor.map(unified_worker_function, param_list_phase1),
                total=len(param_list_phase1),
                desc="Phase 1"
            ))

        # Keep good solutions and learn
        good_lottery_values = {i: [] for i in range(8)}
        good_probabilities = {i: [] for i in range(3)}
        for result in results_phase1:
            if result:
                params, violation = result
                solutions.append(DiscreteSolution(params, violation))
                for i in range(8):
                    good_lottery_values[i].append(int(params[i]))
                for i in range(3):
                    good_probabilities[i].append(params[8 + i])

        print("🎯 Phase 2: Focused exploitation (parallel)")

        # Phase 2: Generate biased parameters
        param_list_phase2 = []
        for _ in range(phase2_attempts):
            params = []
            for i in range(8):
                if random.random() < 0.7 and good_lottery_values[i]:
                    base_val = random.choice(good_lottery_values[i])
                    noise = random.randint(-3, 3)
                    val = np.clip(base_val + noise, self.config.lottery_min, self.config.lottery_max)
                    params.append(float(val))
                else:
                    params.append(float(random.choice(self.lottery_values)))
            for i in range(3):
                if random.random() < 0.7 and good_probabilities[i]:
                    params.append(random.choice(good_probabilities[i]))
                else:
                    params.append(random.choice(self.prob_choices))
            param_list_phase2.append(("exploit", params, config_dict))

        # Run parallel focused evaluation
        with ProcessPoolExecutor(max_workers=self.config.num_cores or mp.cpu_count()) as executor:
            results_phase2 = list(tqdm(
                executor.map(unified_worker_function, param_list_phase2),
                total=len(param_list_phase2),
                desc="Phase 2"
            ))

        for result in results_phase2:
            if result:
                params, violation = result
                solution = DiscreteSolution(params, violation)

                # Check uniqueness
                is_unique = all(not np.allclose(solution.x, existing.x, atol=1e-6) for existing in solutions)
                if is_unique:
                    solutions.append(solution)

        return solutions

    def solve_lottery(self, method: str = 'parallel') -> List[DiscreteSolution]:
        """Main solving method with multiple discrete strategies"""
        print(f"🔧 Starting DISCRETE lottery optimization...")
        print(f"Method: {method}")
        print(f"Search space size: ~{len(self.lottery_values) ** 8 * len(self.prob_choices) ** 3:,}")

        start_time = time.time()

        if method == 'parallel':
            solutions = self.discrete_parallel_search()
        elif method == 'adaptive':
            solutions = self.adaptive_discrete_search()
        elif method == 'random':
            solutions = self.discrete_random_search()
        else:
            raise ValueError(f"Unknown method: {method}")

        end_time = time.time()

        # Sort by violation
        solutions.sort(key=lambda s: s.fun)

        print(f"\n✅ Discrete optimization completed in {end_time - start_time:.2f} seconds")
        print(f"📊 Statistics:")
        print(f"   Total evaluations: {self.stats['evaluations']:,}")
        print(f"   Solutions found: {len(solutions)}")
        if solutions:
            print(f"   Best violation: {min(s.fun for s in solutions):.6f}")
            print(f"   Cache efficiency: {self.stats['cache_hits'] / (self.stats['cache_hits'] + 1):.1%}")

        return solutions

    def save_progress(self, solutions: List[DiscreteSolution], filename: str = None):
        """Save current progress to file"""
        if filename is None:
            filename = f"lottery_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        progress_data = {
            'solutions': [s.to_dict() for s in solutions],
            'stats': self.stats,
            'config': self.config,
            'timestamp': datetime.now()
        }

        with open(filename, 'wb') as f:
            pickle.dump(progress_data, f)

        print(f"💾 Progress saved to: {filename}")
        return filename


    def calculate_solution_diversity(self, solutions: List[DiscreteSolution]) -> Dict[str, float]:
        """Calculate diversity metrics for solution set"""
        if len(solutions) < 2:
            return {'diversity': 0.0, 'num_solutions': len(solutions)}

        # Extract parameter arrays
        params_array = np.array([s.x for s in solutions])

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

    def adaptive_constraint_relaxation(self, initial_threshold: float = None,
                                       min_solutions: int = 5,
                                       max_iterations: int = 5,
                                       relaxation_factor: float = 1.5) -> List[DiscreteSolution]:
        """Gradually relax constraints if too few solutions found"""
        if initial_threshold is None:
            initial_threshold = self.config.violation_threshold

        threshold = initial_threshold
        all_solutions = []
        original_threshold = self.config.violation_threshold

        for iteration in range(max_iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{max_iterations}, threshold: {threshold:.3f}")
            self.config.violation_threshold = threshold

            # Run optimization
            solutions = self.solve_lottery(method='adaptive')
            all_solutions.extend(solutions)

            # Remove duplicates
            unique_solutions = []
            seen = set()
            for sol in all_solutions:
                key = tuple(sol.x)
                if key not in seen:
                    seen.add(key)
                    unique_solutions.append(sol)

            all_solutions = unique_solutions

            print(f"   Total unique solutions so far: {len(all_solutions)}")

            if len(all_solutions) >= min_solutions:
                print(f"✅ Found {len(all_solutions)} solutions, stopping.")
                break

            # Relax threshold
            threshold *= relaxation_factor
            print(f"📈 Only {len(all_solutions)} solutions found, relaxing threshold to {threshold:.3f}...")

        # Restore original threshold
        self.config.violation_threshold = original_threshold

        # Sort by violation
        all_solutions.sort(key=lambda s: s.fun)

        return all_solutions

    def display_solutions(self, solutions: List[DiscreteSolution], save_to_file: bool = True, filename: str = None):
        """Display discrete solutions with comprehensive details"""
        if not solutions:
            message = "No solutions found!"
            print(message)

            # Save "no solutions" message to file if requested
            if save_to_file:
                if filename is None:
                    filename = f"lottery_solutions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(message + "\n")
                print(f"💾 Results saved to: {filename}")

            return []

        # Prepare content for both display and file
        output_lines = []

        # Header
        header = f"\n" + "=" * 80
        output_lines.append(header)
        title = f"FOUND {len(solutions)} LOTTERY SOLUTION(S)"
        output_lines.append(title)
        output_lines.append("=" * 80)
        params_info = f"Prospect Theory Parameters: α={self.alpha}, λ={self.lambda_}, γ={self.gamma}"
        output_lines.append(params_info)

        # Print to console
        print(header)
        print(title)
        print("=" * 80)
        print(params_info)

        all_solution_data = []

        for i, solution in enumerate(solutions):
            params = solution.x
            b11, b12, c21, c22, c31, c32, c33, c34, p1, p2, p3 = params

            # Convert to integers for display
            lottery_values = [int(x) for x in params[:8]]
            b11, b12, c21, c22, c31, c32, c33, c34 = lottery_values

            # Solution header
            solution_header = f"\n--- SOLUTION {i + 1} ---"
            objective_line = f"Objective function value: {solution.fun:.6f}"

            output_lines.append(solution_header)
            output_lines.append(objective_line)
            print(solution_header)
            print(objective_line)

            # Display lottery structure
            structure_header = "LOTTERY STRUCTURE:"
            stage1_line = f"  Stage 1: b₁₁={b11:3d}, b₁₂={b12:3d}"
            stage2_line = f"  Stage 2: c₂₁={c21:3d}, c₂₂={c22:3d}"
            stage3_line = f"  Stage 3: c₃₁={c31:3d}, c₃₂={c32:3d}, c₃₃={c33:3d}, c₃₄={c34:3d}"
            prob_line = f"  Probabilities: p₁={p1:.2f}, p₂={p2:.2f}, p₃={p3:.2f}"

            for line in [structure_header, stage1_line, stage2_line, stage3_line, prob_line]:
                output_lines.append(line)
                print(line)

            # Calculate and display outcomes
            Z, p_array = self.create_lottery_structure(params)
            z1, z2, z3, z4 = Z

            outcomes_header = "FINAL OUTCOMES:"
            z1_line = f"  z₁ = 0+{b11}+{c21}+{c31} = {z1:.0f} (prob={p_array[0]:.3f})"
            z2_line = f"  z₂ = 0+{b11}+{c21}+{c32} = {z2:.0f} (prob={p_array[1]:.3f})"
            z3_line = f"  z₃ = 0+{b12}+{c22}+{c33} = {z3:.0f} (prob={p_array[2]:.3f})"
            z4_line = f"  z₄ = 0+{b12}+{c22}+{c34} = {z4:.0f} (prob={p_array[3]:.3f})"

            for line in [outcomes_header, z1_line, z2_line, z3_line, z4_line]:
                output_lines.append(line)
                print(line)

            # Detailed constraint verification with intervals
            violations, _ = self.check_constraints(params)
            expected_value = np.sum(Z * p_array)
            prob_sum = p1 + p2 + p3

            # Calculate intervals for display
            IL_lower, IL_upper = self.find_monotonic_interval(Z, p_array)
            Z_L1 = np.array([z1, z2])
            p_L1 = np.array([p2, 1 - p2])
            IL1_lower, IL1_upper = self.find_monotonic_interval(Z_L1, p_L1)
            Z_L2 = np.array([z3, z4])
            p_L2 = np.array([p3, 1 - p3])
            IL2_lower, IL2_upper = self.find_monotonic_interval(Z_L2, p_L2)

            E_L1 = z1 * p2 + z2 * (1 - p2)
            E_L2 = z3 * p3 + z4 * (1 - p3)

            # Constraint verification
            constraints_header = "CONSTRAINT VERIFICATION:"
            expected_val_line = f"  1. Expected value: {expected_value:.6f} ≈ 0 ✓" if abs(
                expected_value) < 1 else f"  1. Expected value: {expected_value:.6f} ≠ 0 ✗"
            ordering_line = f"  2. Ordering z1 > 0 >z2 and z3 > 0 > z4: {z1}> 0 >{z2}, {z3}> 0  > {z4} ✓" if \
                (z1 >= 0 >= z2 != z3 and z3 >= 0 >= z4) else f"  2. Ordering constraint ✗"
            output_values_line = f"     Output Lottery Values: 0={0}, b₁₁={b11}, b₁₁+c₂₁={b11 + c21}, b₁₂={b12}, b₁₂+c₂₂={b12 + c22}"
            expectations_line = f"             E(L) = {expected_value:.1f}, E(L₁)={E_L1:.1f}, E(L₂)={E_L2:.1f}"

            for line in [constraints_header, expected_val_line, ordering_line, output_values_line, expectations_line]:
                output_lines.append(line)
                print(line)

            # Interval verification details
            if IL_lower is not None and IL_upper is not None:
                main_interval_line = f"  5. Main interval IL = [{IL_lower:.2f}, {IL_upper:.2f}]"
                values_in_IL = [0, expected_value]
                all_in_IL = all(IL_lower <= v <= IL_upper for v in values_in_IL)
                main_interval_check = f"     Required values: E(L) satisfied ✓" if all_in_IL else f"     Some values outside IL: ✗"

                output_lines.append(main_interval_line)
                output_lines.append(main_interval_check)
                print(main_interval_line)
                print(main_interval_check)

            if IL1_lower is not None and IL1_upper is not None:
                l1_interval_line = f"  6. L1 interval IL1 = [{IL1_lower:.2f}, {IL1_upper:.2f}]"
                values_in_IL1 = [b11, b11 + c21, E_L1, expected_value]
                all_in_IL1 = all(IL1_lower <= v <= IL1_upper for v in values_in_IL1)
                l1_interval_check = f"     Required values: b11, b11+c21, E_L1, EL ✓" if all_in_IL1 else f"     L1 values outside IL1: ✗"

                output_lines.append(l1_interval_line)
                output_lines.append(l1_interval_check)
                print(l1_interval_line)
                print(l1_interval_check)

            if IL2_lower is not None and IL2_upper is not None:
                l2_interval_line = f"  7. L2 interval IL2 = [{IL2_lower:.2f}, {IL2_upper:.2f}]"
                values_in_IL2 = [b12, b12 + c22, E_L2, expected_value]
                all_in_IL2 = all(IL2_lower <= v <= IL2_upper for v in values_in_IL2)
                l2_interval_check = f"     Required values: b12, b12+c22, E_L2, EL ✓" if all_in_IL2 else f"     L2 values outside IL2: ✗"

                output_lines.append(l2_interval_line)
                output_lines.append(l2_interval_check)
                print(l2_interval_line)
                print(l2_interval_check)

            # Store for summary
            all_solution_data.append({
                'Sol': i + 1, 'b11': b11, 'b12': b12, 'c21': c21, 'c22': c22,
                'c31': c31, 'c32': c32, 'c33': c33, 'c34': c34,
                'p1': p1, 'p2': p2, 'p3': p3,
                'z1': int(z1), 'z2': int(z2), 'z3': int(z3), 'z4': int(z4),
                'E[Z]': expected_value, 'Σp': prob_sum, 'Violation': solution.fun
            })

        # Summary table
        summary_header = f"\n" + "=" * 80
        summary_title = "SUMMARY TABLE"
        summary_separator = "=" * 80

        output_lines.append(summary_header)
        output_lines.append(summary_title)
        output_lines.append(summary_separator)

        print(summary_header)
        print(summary_title)
        print(summary_separator)

        df = pd.DataFrame(all_solution_data)
        table_string = df.to_string(index=False, float_format='%.3f')

        # Add table to output and print
        output_lines.append(table_string)
        print(table_string)

        # Calculate and display diversity metrics
        diversity = self.calculate_solution_diversity(solutions)
        diversity_header = f"\n" + "=" * 80
        diversity_title = "SOLUTION DIVERSITY METRICS"
        diversity_separator = "=" * 80

        output_lines.append(diversity_header)
        output_lines.append(diversity_title)
        output_lines.append(diversity_separator)

        print(diversity_header)
        print(diversity_title)
        print(diversity_separator)

        diversity_lines = [
            f"Total solutions: {diversity['num_solutions']}",
            f"Unique lottery combinations: {diversity['unique_lottery_combinations']}",
            f"Unique probability combinations: {diversity['unique_prob_combinations']}",
            f"Mean pairwise distance: {diversity.get('mean_distance', 0):.2f}",
            f"Distance range: [{diversity.get('min_distance', 0):.2f}, {diversity.get('max_distance', 0):.2f}]",
            f"Distance std dev: {diversity.get('std_distance', 0):.2f}"
        ]

        for line in diversity_lines:
            output_lines.append(line)
            print(line)

        # Save to file if requested
        if save_to_file:
            if filename is None:
                # Generate filename with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"lottery_solutions_{timestamp}.txt"

            # Add some metadata at the beginning of the file
            file_header = [
                f"Discrete Lottery Optimization Results",
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Number of solutions found: {len(solutions)}",
                f"Prospect Theory Parameters: α={self.alpha}, λ={self.lambda_}, γ={self.gamma}",
                f"Search Configuration:",
                f"  - Lottery range: [{self.config.lottery_min}, {self.config.lottery_max}]",
                f"  - Probability choices: {self.config.prob_choices}",
                f"  - Violation threshold: {self.config.violation_threshold}",
                f"  - Total evaluations: {self.stats['evaluations']:,}",
                ""  # Empty line
            ]

            try:
                with open(filename, 'w', encoding='utf-8', newline='\n') as f:
                    # Write metadata header
                    for line in file_header:
                        f.write(line + '\n')

                    # Write all the solution details
                    for line in output_lines:
                        f.write(line + '\n')

                    # Add some summary statistics at the end
                    f.write('\n' + '=' * 80 + '\n')
                    f.write('SUMMARY STATISTICS\n')
                    f.write('=' * 80 + '\n')
                    f.write(f"Total solutions found: {len(solutions)}\n")

                    if all_solution_data:
                        violations = [sol['Violation'] for sol in all_solution_data]
                        f.write(f"Best violation score: {min(violations):.6f}\n")
                        f.write(f"Average violation score: {np.mean(violations):.6f}\n")
                        f.write(f"Worst violation score: {max(violations):.6f}\n")

                    f.write(f"\nOptimization efficiency:\n")
                    f.write(f"  Total evaluations: {self.stats['evaluations']:,}\n")
                    f.write(f"  Solutions found: {self.stats['solutions_found']}\n")
                    f.write(f"  Success rate: {self.stats['solutions_found'] / self.stats['evaluations'] * 100:.3f}%\n")
                    f.write(f"  Cache efficiency: {self.stats['cache_hits'] / (self.stats['cache_hits'] + 1):.1%}\n")

                    total_space = len(self.lottery_values) ** 8 * len(self.prob_choices) ** 3
                    searched_fraction = self.stats['evaluations'] / total_space * 100
                    f.write(f"  Search space coverage: {searched_fraction:.2e}%\n")

                    f.write(f"\nFile saved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

                print(f"\n💾 Results saved to: {filename}")

            except Exception as e:
                print(f"\n❌ Error saving file: {e}")
                print("Results displayed on console only.")

        return all_solution_data


def main():
    """Main function for discrete lottery optimization"""

    # Configuration
    config = DiscreteOptimizationConfig(
        alpha=0.88,
        lambda_=2.25,
        gamma=0.61,
        num_attempts=10000000,  # More attempts since it's faster
        violation_threshold=5,
        lottery_min=-100,
        lottery_max=100,
        prob_choices=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        use_local_improvement=True,
        use_adaptive_search=True
    )

    # Create optimizer
    optimizer = DiscreteLotteryOptimizer(config)

    print("🎰 DISCRETE LOTTERY OPTIMIZER")
    print("=" * 50)
    print("Choose optimization method:")
    print("1. Parallel discrete search (recommended)")
    print("2. Adaptive discrete search")
    print("3. Simple random discrete search")
    print("4. Adaptive constraint relaxation")

    # Get user choice
    try:
        choice = input("\nEnter choice (1-4) [default: 2]: ").strip()
        if not choice:
            choice = "2"
        choice = int(choice)
    except:
        choice = 2

    # Run optimization based on choice
    if choice == 1:
        solutions = optimizer.solve_lottery(method='parallel')
    elif choice == 2:
        solutions = optimizer.solve_lottery(method='adaptive')
    elif choice == 3:
        solutions = optimizer.solve_lottery(method='random')
    elif choice == 4:
        print("\n Using adaptive constraint relaxation...")
        solutions = optimizer.adaptive_constraint_relaxation(
            initial_threshold=1.0,
            min_solutions=10,
            max_iterations=5
        )
    else:
        print("Invalid choice, using adaptive search")
        solutions = optimizer.solve_lottery(method='adaptive')

    if solutions:
        # Display results
        solution_data = optimizer.display_solutions(solutions)

        print(f"\n SUCCESS: Found {len(solutions)} discrete solution(s)")
        print(f"Best violation: {min(s.fun for s in solutions):.6f}")

        # Show search efficiency
        total_space = len(optimizer.lottery_values) ** 8 * len(optimizer.prob_choices) ** 3
        searched_fraction = optimizer.stats['evaluations'] / total_space * 100
        print(f"Searched {searched_fraction:.2e}% of total discrete space")

        # Ask if user wants to save progress
        save_progress = input("\nSave progress for future use? (y/n): ").strip().lower()
        if save_progress == 'y':
            optimizer.display_solutions(solutions, save_to_file=True)

        return solutions
    else:
        print("\n No solutions found. Try:")
        print("  - Increasing num_attempts")
        print("  - Relaxing violation_threshold")
        print("  - Using adaptive constraint relaxation (option 4)")
        return None


if __name__ == "__main__":
    solutions = main()