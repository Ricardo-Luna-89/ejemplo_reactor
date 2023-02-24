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

# %% my custom RK4
def RK4(f,t0,x0,h,n,u):
    
    t = np.zeros(n+1)
    x = np.array((n+1)*[x0])
    t[0] = t0
    x[0] = x0
    
    for k in range(n):
        k1 = f(t[k],x[k],u[k])
        k2 = f(t[k]+h/2,x[k]+(h/2)*k1,u[k])
        k3 = f(t[k]+h/2,x[k]+(h/2)*k2,u[k])
        k4 = f(t[k]+h,x[k]+h*k3,u[k])
        x[k+1] = x[k]+(h/6)*(k1+2*k2+2*k3+k4)
        t[k+1] = t[k]+h    
        
    return t,x
# %% condiciones iniciales del modelo e inputs
x0 = np.array([1, 0])
tf = 1.0
u =2.0 # escalar fijo
tspan = (0,tf)
# %% inputs para el custom RK4
n = 20 # número de elementos finitos
h = tf/n # el tamaño del paso de integración
u1 = np.ones(n+1)*u 
# %% integración númerica (scipy)
solution = solve_ivp(reactor_tubular, tspan, x0, method='Radau',
                     args = (u,))

t = solution.t
x = solution.y.T
# %% integración con custom RK4
t1,x1 = RK4(reactor_tubular,0,x0,h,n,u1)
# %%
for i in range(len(x0)): # tamaño de las variables del modelo
    plt.subplot(1,2,i+1)
    plt.plot(t,x[:,i])
    plt.plot(t1,x1[:,i],':')


