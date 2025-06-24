# -*- coding: utf-8 -*-
"""
Created on Wed May 28 14:02:19 2025

@author: XGuo
"""
import simulation_functions as sim

intertemporal_lottery = {
    "time": 0,
    "value": 0,
    "probability": 1.0,
    "next": [
        {
            "time": 1,
            "value": +10,
            "probability": 0.5,
            "next": [
                {
                    "time": 2,
                    "value": 7,
                    "probability": 1.0,
                    "next": [
                        {
                            "time": 3,
                            "value": -7,
                            "probability": 0.33
                        },
                        {
                            "time": 3,
                            "value": -12,
                            "probability": 0.67
                        }
                    ]
                }
            ]
        },
        {
            "time": 1,
            "value": -5,
            "probability": 0.5,
            "next": [
                {
                    "time": 2,
                    "value": 4,
                    "probability": 1,
                    "next" : [
                        {
                            "time": 3,
                            "value": -4,
                            "probability": 0.5
                            },
                        
                        {
                            "time": 3,
                            "value": -9,
                            "probability": 0.5
                            }
                        ]
                }
            ]
        }
    ]
}


paths = sim.generate_paths(intertemporal_lottery)
for i, entry in enumerate(paths):
    print(f"Path {i+1}: {entry['path']}, Probability: {entry['probability']:.4f}")

