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
Z = np.array([30,10,-10,-30])
p = np.array([0.25,0.25,0.25,0.25])




alpha = 0.88
lambda_ = 2.25

sim.plot_ce_comparison(
        param_name='gamma',
        param_start=0.01,
        param_end=1.0,
        num_points=20,
        fixed_params={'alpha': 0.88, 'lambda_': 2.25},
        Z=Z,
        p=p
)



