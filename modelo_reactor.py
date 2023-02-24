# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 12:07:35 2023

@author: Ricardo Luna
"""
# %% import libraries
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# %% model
def reactor_tubular(t,x,u):
    
    dxdt = np.zeros(len(x))
    
    dxdt[0] = -(u+0.5*u**2)*x[0]
    dxdt[1] = u*x[0]
    
    return dxdt
# %% condiciones iniciales del modelo e inputs
x0 = np.array([1, 0])
tf = 1.0
u =2.0
tspan = (0,tf)
# %% integración númerica (scipy)
solution = solve_ivp(reactor_tubular, tspan, x0, method='Radau',
                     args = (u,))

t = solution.t
x = solution.y.T
# %%
plt.subplot(1,2,1)
plt.plot(t,x[:,0])
plt.subplot(1,2,2)
plt.plot(t, x[:,1])

