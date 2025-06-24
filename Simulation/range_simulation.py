# -*- coding: utf-8 -*-
"""
Created on Wed May 28 14:47:40 2025

@author: XGuo
"""


import numpy as np
import matplotlib.pyplot as plt
import simulation_functions as sim
import pandas as pd


# Parameters
r = 0.97
Z = np.array([40,20,-20,-40])
p = np.array([0.25,0.25,0.25,0.25])




alpha = 0.88
lambda_ = 2.25



lambda_set = [1,1.5,2.25,3,3.5,4,4.5]

for lam in lambda_set:
    sim.plot_ce_comparison(
        param_name='gamma',
        param_start=0.01,
        param_end=1.0,
        num_points=20,
        fixed_params={'alpha': 0.88, 'lambda_': lam},
        Z=Z,
        p=p
    )



