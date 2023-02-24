# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 12:07:35 2023

@author: Ricardo Luna
"""
# %% import libraries
import numpy as np
# %% model
def reactor_tubular(t,x,u):
    
    dxdt = np.zeros(len(x))
    
    dxdt[0] = -(u+0.5*u**2)*x[0]
    dxdt[1] = u*x[0]
    
    return dxdt
# %%
x0 = np.array([1, 0])
tf = 1
u =2