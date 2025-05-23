# -*- coding: utf-8 -*-
"""
Created on Fri May 23 11:00:08 2025

@author: XGuo
"""

import numpy as np


# Certainty equivalent function
def ce(alpha, lambda_, gamma, r, Z, p, R):
    def value(z, alpha, lambda_, R):
        diff = z - R
        return np.where(diff >= 0, diff ** alpha, -lambda_ * (-diff) ** alpha)
    
    def weighting(p, gamma):
        numerator = p ** gamma
        denominator = (p ** gamma + (1 - p) ** gamma) ** (1 / gamma)
        return numerator / denominator

    v = value(Z, alpha, lambda_, R)
    w = weighting(p, gamma)
    y = np.sum(v * w)

    if y > 0:
        return R + y ** (1 / alpha)
    else:
        return R - (-y / lambda_) ** (1 / alpha)

