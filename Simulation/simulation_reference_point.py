# -*- coding: utf-8 -*-
"""
Created on Fri May 23 10:59:17 2025

@author: XGuo
"""

import numpy as np
import matplotlib.pyplot as plt
import simulation_functions as sim
# Parameters
r = -0.97
Z = np.array([-1000,-800,-1,0])
p = np.array([0.01,0.1,0.5,0.39])

Z1 = np.array([-400, -150,-1,0])
p1 = np.array([0.01,0.1,0.5,0.39])



alpha = 0.88
lambda_ = 2.25
gamma = 0.69

# Evaluate CE over a range of R values
R_range = np.arange(-1100,1500, 0.01)
CE_values = np.array([sim.ce(alpha, lambda_, gamma, r, Z, p, R) for R in R_range])

CE_values_moderate = np.array([sim.ce(alpha, lambda_, gamma, r, Z1, p1, R) for R in R_range])


# Compute first difference (approximate derivative)
dCE = np.diff(CE_values)
dR = np.diff(R_range)
slope = dCE / dR

# Identify kink locations by detecting large changes in slope
slope_diff = np.abs(np.diff(slope))
threshold = np.percentile(slope_diff, 95)  # top 5% of slope changes
kink_indices = np.where(slope_diff > threshold)[0] + 1  # offset by 1 due to diff

# Extract kink R values
kink_R_values = R_range[kink_indices]

# Plot both CE curves
plt.figure(figsize=(10, 6))
plt.plot(R_range, CE_values, label="CE (Extreme Loss)", color='blue')
plt.plot(R_range, CE_values_moderate, label="CE (Moderate Loss)", color='green')

# Plot kink points for the extreme-loss lottery
plt.plot(kink_R_values, CE_values[kink_indices], 'ro', markersize=2, label="Kinks (Extreme Loss)")

# Labels and legend
plt.xlabel("Reference Point R")
plt.ylabel("Certainty Equivalent")
plt.title("CE Curves for Two Lotteries with Kinks")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Summary output
print("Detected kink positions (Extreme Loss R values):")
print(kink_R_values)
print("Number of kinks (Extreme Loss):", len(kink_R_values))
