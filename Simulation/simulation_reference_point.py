# -*- coding: utf-8 -*-
"""
Created on Fri May 23 10:59:17 2025
@author: XGuo
"""
import numpy as np
import matplotlib.pyplot as plt
import simulation_functions as sim
import pandas as pd

# Parameters
r = 0.97
Z = np.array([-25,15,15,30])
p = np.array([0.25,0.25,0.25,0.25])
alpha = 0.88
lambda_ = 2.25

# TK parameter
gamma = 0.61
l_min = np.min(Z) * 1.5
l_max = np.max(Z) * 1.5

# Evaluate CE over a range of R values
R_range = np.linspace(l_min, l_max, 5000)
y_ce_pairs = [sim.ce(alpha, lambda_, gamma, r, Z, p, R) for R in R_range]
y_values, CE_values = zip(*y_ce_pairs)  # Transpose list of tuples
y_values = np.array(y_values)
CE_values = np.array(CE_values)

# Create DataFrame
ce_df = pd.DataFrame({
    'R': R_range,
    'y': y_values,
    'CE': CE_values
})

R_filtered = ce_df['R'][(ce_df['y'] < 0) & (ce_df['R'] < 20)]
rang = R_filtered.max() - R_filtered.min()

# Compute first difference (approximate derivative)
dCE = np.diff(CE_values)
dR = np.diff(R_range)
slope = dCE / dR

# Identify kink locations by detecting large changes in slope
slope_diff = np.abs(np.diff(slope))
threshold = np.percentile(slope_diff, 99.5)  # top 1% of slope changes
kink_indices = np.where(slope_diff > threshold)[0] + 1  # offset by 1 due to diff

# Extract kink R values
kink_R_values = R_range[kink_indices]

# Find where y_values crosses zero
zero_crossings = []
for i in range(len(y_values) - 1):
    if y_values[i] * y_values[i + 1] < 0:  # Sign change indicates zero crossing
        # Linear interpolation to find more precise crossing point
        R_cross = R_range[i] - y_values[i] * (R_range[i + 1] - R_range[i]) / (y_values[i + 1] - y_values[i])
        zero_crossings.append(R_cross)

# Plot both CE curves
plt.figure(figsize=(10, 6))
plt.plot(R_range, CE_values, label="CE", color='red')
plt.plot(R_range, y_values, label="Value Function", color='green')
ax = plt.gca()

# Plot kink points for the extreme-loss lottery
plt.plot(kink_R_values, CE_values[kink_indices], '.', color="blue", markersize=5, label="Kinks")

# Add vertical lines at zero crossings
for i, zero_cross in enumerate(zero_crossings):
    plt.axvline(x=zero_cross, color='orange', linestyle='--', alpha=0.7, 
                label='Value Function = 0' if i == 0 else "")

# Set y-axis limits BEFORE showing the plot
ax.set_ylim([-30,30])

# Labels and legend
plt.xlabel("Reference Point R")
plt.ylabel("Certainty Equivalent")
plt.title("CE Curves")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Summary output
print("Detected kink positions:")
print(kink_R_values)
print("Number of kinks:", len(kink_R_values))
print("Zero crossings of value function:")
print(zero_crossings)
print("Number of zero crossings:", len(zero_crossings))

# For visualization, compute for a subset of R values
R_range_subset = np.linspace(-30, 30, 500)
values_matrix = []
weights_matrix = []

for R in R_range_subset:
    values = sim.value_function(Z, R, alpha, lambda_)
    weights = sim.weight_function(p, gamma)
    values_matrix.append(values)
    weights_matrix.append(weights)

values_matrix = np.array(values_matrix)  # shape: (len(R), len(Z))
weights_matrix = np.array(weights_matrix)

# Plot individual value function contributions
fig, ax = plt.subplots(figsize=(10, 8))
for i in range(len(Z)):
    ax.plot(R_range_subset, values_matrix[:, i], label=f'v(Z={Z[i]} - R)')

ax.set_xlabel("Reference Point R")
ax.set_ylabel("Value Function Output")
ax.set_title("Value Function Contributions per Outcome")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# Recompute decision weights across R
pi_matrix = []
for R in R_range_subset:
    pi = sim.cumulative_weights(Z, p, R, gamma)
    pi_matrix.append(pi)

pi_matrix = np.array(pi_matrix)

# Plot dynamic decision weights
plt.figure(figsize=(10, 6))
for i in range(len(Z)):
    plt.plot(R_range_subset, pi_matrix[:, i], label=f'π(R) for Z={Z[i]}')

plt.xlabel("Reference Point R")
plt.ylabel("Decision Weight π_i(R)")
plt.title("Dynamic Rank-Dependent Decision Weights")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()