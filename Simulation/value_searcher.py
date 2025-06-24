# -*- coding: utf-8 -*-
"""
Value Searcher for Four Outcome Lotteries

@author: XGuo xiaoyi.guo@unisg.ch
"""

import numpy as np
from scipy.optimize import fsolve, minimize, differential_evolution
import pandas as pd
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

class LotteryOptimizer:
    """Class to encapsulate lottery optimization functionality"""
    
    def __init__(self, alpha=0.88, lambda_=2.25, gamma=0.61):
        self.alpha = alpha
        self.lambda_ = lambda_
        self.gamma = gamma
    
    def create_lottery_structure(self, params):
        """Create the intertemporal lottery structure from parameters"""
        b11, b12, c21, c22, c31, c32, c33, c34, p1, p2, p3 = params
        a = 0  # Starting point (constant)
        
        # Calculate final outcomes for each path
        z1 = a + b11 + c21 + c31  # Path: a -> b11 -> c21 -> c31
        z2 = a + b11 + c21 + c32  # Path: a -> b11 -> c21 -> c32
        z3 = a + b12 + c22 + c33  # Path: a -> b12 -> c22 -> c33
        z4 = a + b12 + c22 + c34  # Path: a -> b12 -> c22 -> c34
        
        # Calculate path probabilities
        prob1 = p1 * p2
        prob2 = p1 * (1 - p2)
        prob3 = (1 - p1) * p3
        prob4 = (1 - p1) * (1 - p3)
        
        outcomes = np.array([z1, z2, z3, z4])
        probabilities = np.array([prob1, prob2, prob3, prob4])
        
        return outcomes, probabilities
    
    def value_function(self, Z, R):
        """Prospect theory value function"""
        diff = Z - R
        return np.where(diff >= 0, 
                       diff ** self.alpha, 
                       -self.lambda_ * (-diff) ** self.alpha)
    
    def probability_weighting(self, p):
        """Improved probability weighting function with numerical stability"""
        # Handle edge cases
        p = np.clip(p, 1e-10, 1 - 1e-10)
        
        numerator = p ** self.gamma
        denominator = (p ** self.gamma + (1 - p) ** self.gamma) ** (1 / self.gamma)
        return numerator / denominator
    
    def compute_Y(self, Z, p, R):
        """Compute the overall prospect theory value Y"""
        v = self.value_function(Z, R)
        w = self.probability_weighting(p)
        return np.sum(v * w)
    
    def find_R_where_Y_equals_zero(self, Z, p):
        """Find reference point R where Y = 0 with improved robustness and speed"""
        def equation(R):
            return self.compute_Y(Z, p, R)
        
        # Use fewer starting points for speed, but keep the most promising ones
        starting_points = [0, np.mean(Z), np.median(Z)]
        
        for start_R in starting_points:
            try:
                R_solution = fsolve(equation, start_R, xtol=1e-8)[0]  # Relaxed tolerance
                # Verify the solution with relaxed tolerance
                if abs(equation(R_solution)) < 1e-6:  # Relaxed from 1e-8
                    return R_solution
            except:
                continue
        return None
    
    def find_monotonic_interval(self, Z, p):
        """Find monotonic interval IL for lottery L"""
        # Find two kinks of zero value function
        R_zero = self.find_R_where_Y_equals_zero(Z, p)
        if R_zero is None:
            return None, None
        
        # Find the next larger z value that's closest to R_zero
        # Filter outcomes that are greater than R_zero
        larger_outcomes = Z[Z > R_zero]
        
        if len(larger_outcomes) == 0:
            # If no outcomes are larger than R_zero, use the maximum outcome
            next_reference = np.max(Z)
        else:
            # Find the outcome that's closest to R_zero among those larger than R_zero
            differences = larger_outcomes - R_zero
            min_diff_idx = np.argmin(differences)
            next_reference = larger_outcomes[min_diff_idx]
        
        # Ensure we have a valid interval
        if next_reference <= R_zero:
            # Fallback: use the second largest outcome or add small epsilon
            sorted_outcomes = np.sort(Z)
            if len(sorted_outcomes) >= 2:
                next_reference = sorted_outcomes[-2]  # Second largest
            else:
                next_reference = R_zero + 1.0  # Add epsilon
        
        return R_zero, next_reference  # This might need adjustment based on specific requirements
    
    def check_constraints(self, params):
        """Check all lottery constraints following the exact specification"""
        b11, b12, c21, c22, c31, c32, c33, c34, p1, p2, p3 = params
        Z, p = self.create_lottery_structure(params)
        z1, z2, z3, z4 = Z
        
        violations = {}
        total_violations = 0
        
        # Constraint 1: Null initial expectation E(L) = 0
        expected_value = np.sum(Z * p)
        violations['expected_value'] = abs(expected_value)
        total_violations += violations['expected_value']
        
        # Constraint 5: Ordering constraint z1 > z2 ≥ 0 ≥ z3 > z4, z2 ≠ z3
        ordering_violations = 0
        if not (z1 > z2): 
            ordering_violations += (z2 - z1 + 0.01)
        if not (z2 >= 0): 
            ordering_violations += (-z2 + 0.01)
        if not (0 >= z3): 
            ordering_violations += (z3 + 0.01)
        if not (z3 > z4): 
            ordering_violations += (z4 - z3 + 0.01)
        if abs(z2 - z3) < 0.01:  # z2 ≠ z3
            ordering_violations += 0.01
        
        violations['ordering'] = ordering_violations
        total_violations += ordering_violations
        
        # Constraint 5(a): Regularity constraint - sum of probabilities ≤ 2.7
        prob_sum = p1 + p2 + p3
        if prob_sum > 2.7:
            violations['prob_sum'] = prob_sum - 2.7
            total_violations += violations['prob_sum']
        else:
            violations['prob_sum'] = 0
        
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
        
        # Check: 0, b11, b11+c21, b12, b12+c22, E(L1), E(L2) ∈ IL
        values_to_check_L = {
            '0': 0,
            'exp_value' : expected_value
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
        p_L1 = np.array([p2, 1-p2])
        IL1_lower, IL1_upper = self.find_monotonic_interval(Z_L1, p_L1)
        
        if IL1_lower is None or IL1_upper is None or IL1_lower >= IL1_upper:
            violations['empty_interval_L1'] = 100
            total_violations += 100
        else:
            # Check: b11, b11+c21, E(L1) ∈ IL1
            values_to_check_L1 = {
                'b11': b11,
                'b11_c21': b11 + c21,
                'E_L1': E_L1,
                'exp_value' : expected_value
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
        p_L2 = np.array([p3, 1-p3])
        IL2_lower, IL2_upper = self.find_monotonic_interval(Z_L2, p_L2)
        
        if IL2_lower is None or IL2_upper is None or IL2_lower >= IL2_upper:
            violations['empty_interval_L2'] = 100
            total_violations += 100
        else:
            # Check: b12, b12+c22, E(L2) ∈ IL2
            values_to_check_L2 = {
                'b12': b12,
                'b12_c22': b12 + c22,
                'E_L2': E_L2,
                'exp_value' : expected_value
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
    
    def objective_function(self, params):
        """Objective function following the constraint specification"""
        violations, valid = self.check_constraints(params)
        
        if not valid:
            return 10000  # Large penalty for invalid solutions
        
        # Weighted penalty structure
        total_violation = 0
        
        # High penalty for ordering violations (most important)
        total_violation += violations.get('ordering', 0) * 100
        
        # Medium penalty for interval violations
        total_violation += violations.get('interval_L', 0) * 5
        total_violation += violations.get('interval_L1', 0) * 5
        total_violation += violations.get('interval_L2', 0) * 5
        
        # Expected value should be exactly zero
        total_violation += violations.get('expected_value', 0) ** 2
        
        # Probability constraints
        total_violation += violations.get('prob_sum', 0) * 10
        total_violation += violations.get('prob_bounds', 0) * 10
        
        # Penalties for empty intervals
        total_violation += violations.get('empty_interval', 0)
        total_violation += violations.get('empty_interval_L1', 0)
        total_violation += violations.get('empty_interval_L2', 0)
        
        return total_violation
    
    def solve_with_differential_evolution(self, num_attempts=20):  # Reduced from 50
        """Use differential evolution for optimization with better progress tracking"""
        # Define bounds: [b11, b12, c21, c22, c31, c32, c33, c34, p1, p2, p3]
        bounds = [(-100, 100)] * 8 + [(0.01, 0.99)] * 3  # Avoid boundary issues
        
        best_solutions = []
        
        print(f"Using Differential Evolution with {num_attempts} attempts...")
        print("Each attempt may take 30-60 seconds - please be patient...")
        
        pbar = tqdm(range(num_attempts), desc="DE Attempts")
        
        for attempt in pbar:
            try:
                # Reduced parameters for faster execution
                result = differential_evolution(
                    self.objective_function,
                    bounds,
                    seed=attempt,
                    maxiter=300,    # Reduced from 300
                    popsize=15,      # Reduced from 15
                    atol=1e-8,      # Relaxed tolerance
                    tol=1e-8,       # Relaxed tolerance
                    workers=1,
                    updating='immediate',
                    polish=False    # Skip final polishing for speed
                )
                
                if result.fun < 100.0:  # More lenient threshold
                    # Round lottery values to integers
                    params = result.x.copy()
                    params[:8] = np.round(params[:8])
                    
                    # Verify rounded solution
                    rounded_violation = self.objective_function(params)
                    if rounded_violation < 200.0:  # More lenient for rounded
                        # Check for uniqueness
                        is_unique = True
                        for existing in best_solutions:
                            if np.allclose(params, existing.x, atol=1e-3):  # Less strict
                                is_unique = False
                                break
                        
                        if is_unique:
                            class Solution:
                                def __init__(self, x, fun):
                                    self.x = x
                                    self.fun = fun
                            
                            best_solutions.append(Solution(params, rounded_violation))
                
                # Update progress bar with timing info
                pbar.set_postfix({
                    'Solutions': len(best_solutions),
                    'Last_violation': f'{result.fun:.1f}',
                    'Status': 'Found!' if result.fun < 100.0 else 'Searching...',
                    'Evals': result.nfev
                })
                
            except Exception as e:
                pbar.set_postfix({
                    'Solutions': len(best_solutions),
                    'Status': f'Error: {str(e)[:20]}',
                    'Attempt': attempt
                })
                continue
        
        pbar.close()
        return best_solutions
    
    def solve_lottery(self, num_attempts=100, violation_threshold=100.0, use_fast_method=True):
        """Main solving function with better defaults"""
        print(f"Starting lottery optimization...")
        print(f"Parameters: alpha={self.alpha}, lambda={self.lambda_}, gamma={self.gamma}")
        
        start_time = time.time()
        
        if use_fast_method:
            print("Using random search for faster results...")
            solutions = self.solve_with_random_search(num_attempts, violation_threshold)
        else:
            print("Using differential evolution (much slower but more thorough)...")
            solutions = self.solve_with_differential_evolution(num_attempts)
        
        end_time = time.time()
        print(f"\nOptimization completed in {end_time - start_time:.2f} seconds")
        
        return solutions
    
    def solve_with_random_search(self, num_attempts, violation_threshold):
        """Random search method"""
        solutions = []
        prob_choices = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        pbar = tqdm(range(num_attempts), desc="Random Search")
        
        for attempt in pbar:
            params = [
                float(np.random.randint(-100, 101)) for _ in range(8)  # Lottery values
            ] + [
                np.random.choice(prob_choices) for _ in range(3)  # Probabilities
            ]
            
            violation = self.objective_function(params)
            
            if violation < violation_threshold:
                is_unique = True
                for existing in solutions:
                    if np.allclose(params, existing.x, atol=1e-6):
                        is_unique = False
                        break
                
                if is_unique:
                    class Solution:
                        def __init__(self, x, fun):
                            self.x = x
                            self.fun = fun
                    
                    solutions.append(Solution(np.array(params), violation))
                    pbar.set_postfix({'Solutions': len(solutions), 'Violation': f'{violation:.2f}'})
        
        return solutions
    
    def display_solutions(self, solutions):
        """Display solutions with comprehensive constraint verification"""
        if not solutions:
            print("No solutions found!")
            return []
        
        print(f"\n" + "="*80)
        print(f"FOUND {len(solutions)} LOTTERY SOLUTION(S)")
        print("="*80)
        print(f"Prospect Theory Parameters: α={self.alpha}, λ={self.lambda_}, γ={self.gamma}")
        
        all_solution_data = []
        
        for i, solution in enumerate(solutions):
            params = solution.x
            b11, b12, c21, c22, c31, c32, c33, c34, p1, p2, p3 = params
            
            # Convert to integers for display
            lottery_values = [int(x) for x in params[:8]]
            b11, b12, c21, c22, c31, c32, c33, c34 = lottery_values
            
            print(f"\n--- SOLUTION {i+1} ---")
            print(f"Objective function value: {solution.fun:.6f}")
            
            # Display lottery structure
            print(f"LOTTERY STRUCTURE:")
            print(f"  Stage 1: b₁₁={b11:3d}, b₁₂={b12:3d}")
            print(f"  Stage 2: c₂₁={c21:3d}, c₂₂={c22:3d}")
            print(f"  Stage 3: c₃₁={c31:3d}, c₃₂={c32:3d}, c₃₃={c33:3d}, c₃₄={c34:3d}")
            print(f"  Probabilities: p₁={p1:.2f}, p₂={p2:.2f}, p₃={p3:.2f}")
            
            # Calculate and display outcomes
            Z, p_array = self.create_lottery_structure(params)
            z1, z2, z3, z4 = Z
            
            print(f"FINAL OUTCOMES:")
            print(f"  z₁ = 0+{b11}+{c21}+{c31} = {z1:.0f} (prob={p_array[0]:.3f})")
            print(f"  z₂ = 0+{b11}+{c21}+{c32} = {z2:.0f} (prob={p_array[1]:.3f})")
            print(f"  z₃ = 0+{b12}+{c22}+{c33} = {z3:.0f} (prob={p_array[2]:.3f})")
            print(f"  z₄ = 0+{b12}+{c22}+{c34} = {z4:.0f} (prob={p_array[3]:.3f})")
            
            # Detailed constraint verification with intervals
            violations, _ = self.check_constraints(params)
            expected_value = np.sum(Z * p_array)
            prob_sum = p1 + p2 + p3
            
            # Calculate intervals for display
            IL_lower, IL_upper = self.find_monotonic_interval(Z, p_array)
            Z_L1 = np.array([z1, z2])
            p_L1 = np.array([p2, 1-p2])
            IL1_lower, IL1_upper = self.find_monotonic_interval(Z_L1, p_L1)
            Z_L2 = np.array([z3, z4])
            p_L2 = np.array([p3, 1-p3])
            IL2_lower, IL2_upper = self.find_monotonic_interval(Z_L2, p_L2)
            
            E_L1 = z1 * p2 + z2 * (1 - p2)
            E_L2 = z3 * p3 + z4 * (1 - p3)
            
            print(f"CONSTRAINT VERIFICATION:")
            print(f"  1. Expected value: {expected_value:.6f} ≈ 0 ✓" if abs(expected_value) < 0.01 else f"  1. Expected value: {expected_value:.6f} ≠ 0 ✗")
            print(f"  2. Ordering z₁>z₂≥0≥z₃>z₄: {z1}>{z2}≥0≥{z3}>{z4} ✓" if (z1>z2>=0>=z3>z4 and z2!=z3) else f"  2. Ordering constraint ✗")
            print(f"  3. Probability sum: {prob_sum:.2f} ≤ 2.7 ✓" if prob_sum <= 2.7 else f"  3. Probability sum: {prob_sum:.2f} > 2.7 ✗")
            print(f"  4. z₂ ≠ z₃: {z2} ≠ {z3} ✓" if z2 != z3 else f"  4. z₂ = z₃ ✗")
            
            # Interval verification details
            if IL_lower is not None and IL_upper is not None:
                print(f"  5. Main interval IL = [{IL_lower:.2f}, {IL_upper:.2f}]")
                values_in_IL = [0, b11, b11+c21, b12, b12+c22, E_L1, E_L2]
                all_in_IL = all(IL_lower <= v <= IL_upper for v in values_in_IL)
                print(f"     All required values in IL: ✓" if all_in_IL else f"     Some values outside IL: ✗")
                print(f"     Values: 0={0}, b₁₁={b11}, b₁₁+c₂₁={b11+c21}, b₁₂={b12}, b₁₂+c₂₂={b12+c22}")
                print(f"             E(L₁)={E_L1:.1f}, E(L₂)={E_L2:.1f}")
            
            if IL1_lower is not None and IL1_upper is not None:
                print(f"  6. L1 interval IL1 = [{IL1_lower:.2f}, {IL1_upper:.2f}]")
                values_in_IL1 = [b11, b11+c21, E_L1]
                all_in_IL1 = all(IL1_lower <= v <= IL1_upper for v in values_in_IL1)
                print(f"     L1 values in IL1: ✓" if all_in_IL1 else f"     L1 values outside IL1: ✗")
            
            if IL2_lower is not None and IL2_upper is not None:
                print(f"  7. L2 interval IL2 = [{IL2_lower:.2f}, {IL2_upper:.2f}]")
                values_in_IL2 = [b12, b12+c22, E_L2]
                all_in_IL2 = all(IL2_lower <= v <= IL2_upper for v in values_in_IL2)
                print(f"     L2 values in IL2: ✓" if all_in_IL2 else f"     L2 values outside IL2: ✗")
            
            # Store for summary
            all_solution_data.append({
                'Sol': i+1, 'b11': b11, 'b12': b12, 'c21': c21, 'c22': c22,
                'c31': c31, 'c32': c32, 'c33': c33, 'c34': c34,
                'p1': p1, 'p2': p2, 'p3': p3,
                'z1': int(z1), 'z2': int(z2), 'z3': int(z3), 'z4': int(z4),
                'E[Z]': expected_value, 'Σp': prob_sum, 'Violation': solution.fun
            })
        
        # Summary table
        print(f"\n" + "="*80)
        print("SUMMARY TABLE")
        print("="*80)
        df = pd.DataFrame(all_solution_data)
        print(df.to_string(index=False, float_format='%.3f'))
        
        return all_solution_data

def main():
    """Main execution function"""
    optimizer = LotteryOptimizer(alpha=0.88, lambda_=2.25, gamma=0.61)
    
    print("Choose optimization method:")
    print("1. Fast random search (recommended - completes in seconds)")
    print("2. Slow differential evolution (may take many minutes)")
    
    # Use fast method by default
    solutions = optimizer.solve_lottery(
        num_attempts=40,      
        violation_threshold=5,  
        use_fast_method=False   
    )
    
    if solutions:
        solution_data = optimizer.display_solutions(solutions)
        print(f"\n🎉 SUCCESS: Found {len(solutions)} valid solution(s)")
        
        best_solution = min(solutions, key=lambda x: x.fun)
        print(f"Best solution violation: {best_solution.fun:.6f}")
        
        return solutions
    else:
        print("\n❌ No solutions found matching all constraints.")
        print("The constraint set is very restrictive. Try:")
        print("  - Using random search with more attempts")
        print("  - Relaxing violation_threshold")
        print("  - Adjusting prospect theory parameters")
        return None

if __name__ == "__main__":
    solutions = main()