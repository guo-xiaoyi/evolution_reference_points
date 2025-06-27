# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 22:14:36 2025

@author: XGuo
"""


# -*- coding: utf-8 -*-
"""
Pure Discrete Lottery Optimizer - Converted from Continuous DE

Key changes:
1. Direct discrete parameter generation (no rounding)
2. Multiple discrete optimization methods
3. Eliminated all continuous → discrete conversion issues
4. Much faster and more reliable
5. Better suited for your constraint structure

@author: XGuo xiaoyi.guo@unisg.ch (original)
@converted_by: Claude (discrete optimization)
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
warnings.filterwarnings('ignore')

@dataclass
class DiscreteOptimizationConfig:
    """Configuration for discrete optimization"""
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

def discrete_worker_function(args):
    """Worker function for parallel discrete optimization"""
    try:
        seed, violation_threshold, config_dict = args
        
        # Set seed for reproducibility
        np.random.seed(seed)
        random.seed(seed)
        
        # Recreate config and optimizer
        config = DiscreteOptimizationConfig(**config_dict)
        optimizer = DiscreteLotteryOptimizer(config)
        
        # Generate random discrete parameters
        params = optimizer.generate_random_discrete_params()
        
        # Evaluate
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
        return {'error': str(e), 'seed': seed}

class DiscreteLotteryOptimizer:
    """Pure discrete lottery optimizer - no continuous variables!"""
    
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
            'start_time': None
        }
    
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
    
    def probability_weighting(self, p: np.ndarray) -> np.ndarray:
        """Probability weighting function with numerical stability"""
        p = np.clip(p, 1e-10, 1 - 1e-10)
        
        numerator = np.power(p, self.gamma)
        denominator = np.power(np.power(p, self.gamma) + np.power(1 - p, self.gamma), 1 / self.gamma)
        return numerator / denominator
    
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
        """Check all constraints - same logic as original"""
        try:
            self.stats['evaluations'] += 1
            
            # Extract parameters  
            b11, b12, c21, c22, c31, c32, c33, c34, p1, p2, p3 = params
            
            # Basic validation
            for p in [p1, p2, p3]:
                if not (0 < p < 1):
                    return {'total': 10000}, False
            
            Z, p_array = self.create_lottery_structure(params)
            z1, z2, z3, z4 = Z
            
            violations = {}
            total_violations = 0
            
            # Constraint 1: Null expectation
            expected_value = float(np.sum(Z * p_array))
            violations['expected_value'] = abs(expected_value)
            total_violations += violations['expected_value']
            
            # Constraint 2: Ordering
            ordering_violations = 0
            if not (z1 > z2): ordering_violations += (z2 - z1 + 1)
            if not (z2 >= 0): ordering_violations += (-z2 + 1) if z2 < 0 else 0
            if not (z3 <= 0): ordering_violations += (z3 + 1) if z3 > 0 else 0
            if not (z3 > z4): ordering_violations += (z4 - z3 + 1)
            if abs(z2 - z3) < 0.01: ordering_violations += 2
            
            violations['ordering'] = ordering_violations
            total_violations += ordering_violations
            
            # Early exit for major violations
            if ordering_violations > 10:
                violations['total'] = total_violations
                return violations, False
            
            # Interval constraints (simplified for speed)
            IL_lower, IL_upper = self.find_monotonic_interval(Z, p_array)
            if IL_lower is None:
                violations['interval'] = 1000
                total_violations += 1000
            else:
                interval_violations = 0
                for value in [0, expected_value]:
                    if value < IL_lower:
                        interval_violations += (IL_lower - value)
                    elif value > IL_upper:
                        interval_violations += (value - IL_upper)
                violations['interval'] = interval_violations
                total_violations += interval_violations
            
            violations['total'] = total_violations
            return violations, True
            
        except Exception:
            return {'total': 10000}, False
    
    def objective_function(self, params: List[float]) -> float:
        """Objective function with weighted penalties"""
        violations, valid = self.check_constraints(params)
        
        if not valid:
            return 10000.0
        
        # Weighted penalties
        total_violation = (
            violations.get('ordering', 0) * 100 +
            violations.get('expected_value', 0) * 50 +
            violations.get('interval', 0) * 10
        )
        
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
            (seed, self.config.violation_threshold, config_dict)
            for seed in range(self.config.num_attempts)
        ]
        
        print(f"🚀 Running parallel discrete search with {num_cores} cores...")
        
        solutions = []
        self.stats['start_time'] = time.time()
        
        # Process in batches
        batch_size = 500
        
        with tqdm(total=self.config.num_attempts, desc="Parallel Discrete") as pbar:
            for i in range(0, len(args_list), batch_size):
                batch_args = args_list[i:i+batch_size]
                
                with ProcessPoolExecutor(max_workers=num_cores) as executor:
                    batch_results = list(executor.map(discrete_worker_function, batch_args))
                
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
                    'Rate': f'{len(batch_args)/(time.time() - self.stats["start_time"] + 1e-6):.0f}/s'
                })
        
        return solutions
    
    def adaptive_discrete_search(self) -> List[DiscreteSolution]:
        """Adaptive search that learns from good solutions"""
        solutions = []
        
        # Phase 1: Random exploration
        print("📊 Phase 1: Random exploration...")
        phase1_attempts = self.config.num_attempts // 3
        
        good_lottery_values = {i: [] for i in range(8)}
        good_probabilities = {i: [] for i in range(3)}
        
        for attempt in tqdm(range(phase1_attempts), desc="Exploration"):
            params = self.generate_random_discrete_params()
            violation = self.objective_function(params)
            
            if violation < self.config.violation_threshold * 2:  # More lenient for learning
                solution = DiscreteSolution(params, violation)
                solutions.append(solution)
                
                # Learn from good solutions
                for i in range(8):
                    good_lottery_values[i].append(int(params[i]))
                for i in range(3):
                    good_probabilities[i].append(params[8 + i])
        
        # Phase 2: Focused search
        print("🎯 Phase 2: Focused search...")
        phase2_attempts = self.config.num_attempts - phase1_attempts
        
        for attempt in tqdm(range(phase2_attempts), desc="Focused"):
            # Generate params biased toward good regions
            params = []
            
            # Lottery values: 70% from good regions, 30% random
            for i in range(8):
                if random.random() < 0.7 and good_lottery_values[i]:
                    # Sample from good values with some noise
                    base_val = random.choice(good_lottery_values[i])
                    noise = random.randint(-3, 3)
                    val = np.clip(base_val + noise, self.config.lottery_min, self.config.lottery_max)
                    params.append(float(val))
                else:
                    params.append(float(random.choice(self.lottery_values)))
            
            # Probabilities: 70% from good regions, 30% random
            for i in range(3):
                if random.random() < 0.7 and good_probabilities[i]:
                    params.append(random.choice(good_probabilities[i]))
                else:
                    params.append(random.choice(self.prob_choices))
            
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
                    solutions.append(solution)
        
        return solutions
    
    def solve_lottery(self, method: str = 'parallel') -> List[DiscreteSolution]:
        """Main solving method with multiple discrete strategies"""
        print(f"🎲 Starting DISCRETE lottery optimization...")
        print(f"Method: {method}")
        print(f"Search space size: ~{len(self.lottery_values)**8 * len(self.prob_choices)**3:,}")
        
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
        print(f"   Success rate: {len(solutions)/self.stats['evaluations']*100:.4f}%")
        print(f"   Best violation: {min(s.fun for s in solutions) if solutions else 'N/A'}")
        
        return solutions
    
    def display_solutions(self, solutions: List[DiscreteSolution], save_to_file: bool = True, filename: str = None):
        """Display discrete solutions with comprehensive details"""
        if not solutions:
            print("❌ No discrete solutions found!")
            return []
        
        print(f"\n{'='*80}")
        print(f"FOUND {len(solutions)} DISCRETE LOTTERY SOLUTION(S)")
        print(f"{'='*80}")
        print(f"Prospect Theory Parameters: α={self.alpha}, λ={self.lambda_}, γ={self.gamma}")
        
        solution_data = []
        
        for i, solution in enumerate(solutions[:10]):  # Show top 10
            print(f"\n--- DISCRETE SOLUTION {i+1} ---")
            print(f"Objective function value: {solution.fun:.6f}")
            
            # Display lottery structure
            b11, b12, c21, c22, c31, c32, c33, c34 = solution.lottery_values
            p1, p2, p3 = solution.probabilities
            
            print("LOTTERY STRUCTURE:")
            print(f"  Stage 1: b₁₁={b11:3d}, b₁₂={b12:3d}")
            print(f"  Stage 2: c₂₁={c21:3d}, c₂₂={c22:3d}")
            print(f"  Stage 3: c₃₁={c31:3d}, c₃₂={c32:3d}, c₃₃={c33:3d}, c₃₄={c34:3d}")
            print(f"  Probabilities: p₁={p1:.1f}, p₂={p2:.1f}, p₃={p3:.1f}")
            
            # Calculate outcomes
            Z, p_array = self.create_lottery_structure(solution.x.tolist())
            z1, z2, z3, z4 = Z
            expected_value = np.sum(Z * p_array)
            
            print("FINAL OUTCOMES:")
            print(f"  z₁ = 0+{b11}+{c21}+{c31} = {z1:.0f} (prob={p_array[0]:.3f})")
            print(f"  z₂ = 0+{b11}+{c21}+{c32} = {z2:.0f} (prob={p_array[1]:.3f})")
            print(f"  z₃ = 0+{b12}+{c22}+{c33} = {z3:.0f} (prob={p_array[2]:.3f})")
            print(f"  z₄ = 0+{b12}+{c22}+{c34} = {z4:.0f} (prob={p_array[3]:.3f})")
            print(f"  Expected value: {expected_value:.6f}")
            
            # Store for summary
            solution_data.append({
                'Sol': i+1,
                'b11': b11, 'b12': b12, 'c21': c21, 'c22': c22,
                'c31': c31, 'c32': c32, 'c33': c33, 'c34': c34,
                'p1': p1, 'p2': p2, 'p3': p3,
                'z1': int(z1), 'z2': int(z2), 'z3': int(z3), 'z4': int(z4),
                'E[Z]': expected_value,
                'Violation': solution.fun
            })
        
        # Summary table
        print(f"\n{'='*80}")
        print("SUMMARY TABLE")
        print(f"{'='*80}")
        
        df = pd.DataFrame(solution_data)
        print(df.to_string(index=False, float_format='%.3f'))
        
        # Save to file
        if save_to_file:
            if filename is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"discrete_lottery_solutions_{timestamp}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Discrete Lottery Optimization Results\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Solutions: {len(solutions)}\n")
                f.write(f"Parameters: α={self.alpha}, λ={self.lambda_}, γ={self.gamma}\n\n")
                f.write(df.to_string(index=False, float_format='%.6f'))
            
            print(f"\n💾 Results saved to: {filename}")
        
        return solution_data

def main():
    """Main function for discrete lottery optimization"""
    
    # Configuration
    config = DiscreteOptimizationConfig(
        alpha=0.88,
        lambda_=2.25, 
        gamma=0.61,
        num_attempts=5000,          # More attempts since it's faster
        violation_threshold=5.0,
        lottery_min=-50,
        lottery_max=50,
        prob_choices=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        use_local_improvement=True
    )
    
    # Create optimizer
    optimizer = DiscreteLotteryOptimizer(config)
    
    print("🎲 DISCRETE LOTTERY OPTIMIZER")
    print("="*50)
    print("Choose optimization method:")
    print("1. Parallel discrete search (recommended)")
    print("2. Adaptive discrete search")  
    print("3. Simple random discrete search")
    
    # Run optimization
    solutions = optimizer.solve_lottery(method='parallel')
    
    if solutions:
        # Display results
        solution_data = optimizer.display_solutions(solutions)
        
        print(f"\n🎉 SUCCESS: Found {len(solutions)} discrete solution(s)")
        print(f"Best violation: {min(s.fun for s in solutions):.6f}")
        
        # Show search efficiency
        total_space = len(optimizer.lottery_values)**8 * len(optimizer.prob_choices)**3
        searched_fraction = optimizer.stats['evaluations'] / total_space * 100
        print(f"Searched {searched_fraction:.2e}% of total discrete space")
        
        return solutions
    else:
        print("\n❌ No solutions found. Try:")
        print("  - Increasing num_attempts")
        print("  - Relaxing violation_threshold") 
        print("  - Using adaptive search method")
        return None

if __name__ == "__main__":
    solutions = main()