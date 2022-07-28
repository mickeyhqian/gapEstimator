#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 22:34:21 2022

@author: mickey
"""

import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
import gurobipy as gp


# CVaR problem
def cvarSAA(data, x0=None):
    n = len(data)
    alpha = 0.1
    data = sorted(data)
    idx = int(np.ceil(alpha * n))
    x = data[-idx]
    SAA_cost = x + np.maximum(data - x, 0) / alpha
    if x0 is not None:
        SAA_cost -= x0 + np.maximum(data - x0, 0) / alpha
    return x, np.mean(SAA_cost), np.var(SAA_cost) # SAA solution, SAA objective value, SAA objective variance

def cvarSAAObj(data, x, x0=None):
    alpha = 0.1
    SAA_cost = x + np.maximum(data - x, 0) / alpha
    if x0 is not None:
        SAA_cost -= x0 + np.maximum(data - x0, 0) / alpha
    return np.mean(SAA_cost), np.var(SAA_cost)
    
def cvarObj(x):
    alpha = 0.1
    def func(t):
        return 1 / np.sqrt(2 * np.pi) * np.exp(-t**2 / 2) * (t - x)
    return x + quad(func, x, np.inf)[0] / alpha

def cvarSample(n, rng):
    return rng.normal(size=n)

def cvarSol():
    return norm.ppf(0.9)


# Portfolio problem
rng = np.random.default_rng(seed=1235)
portSigmaRoot = rng.uniform(size=(5, 5))
portSigma = np.matmul(portSigmaRoot.T, portSigmaRoot)
portMu = np.arange(1, 6)
portB = 3
def portfolioSAA(data, x0=None):
    alpha = 0.05
    n = len(data)
    m = gp.Model()
    m.Params.OutputFlag = 0
    x = m.addVars(5)
    c = m.addVar(lb=-float("inf"), obj=1)
    s = m.addVars(n, obj=1/(n*alpha))
    m.addConstrs(gp.quicksum([-data[i, j]*x[j] for j in range(len(x))]) - c <= s[i] for i in range(len(s)))
    m.addConstr(gp.quicksum([portMu[i]*x[i] for i in range(len(x))]) >= portB)
    m.addConstr(gp.quicksum(x) == 1)
    m.optimize()
    xval = np.array(m.x[:5])
    cval = m.x[5]
    SAA_cost = cval + np.maximum(-data.dot(xval) - cval, 0) / alpha
    if x0 is not None:
        SAA_cost -= x0[5] + np.maximum(-data.dot(x0[:5]) - x0[5], 0) / alpha
    return np.concatenate((xval, [cval])), np.mean(SAA_cost), np.var(SAA_cost)

def portfolioSAAObj(data, x, x0=None):
    alpha = 0.05
    xval = np.array(x[:5])
    cval = x[5]
    SAA_cost = cval + np.maximum(-data.dot(xval) - cval, 0) / alpha
    if x0 is not None:
        SAA_cost -= x0[5] + np.maximum(-data.dot(x0[:5]) - x[5], 0) / alpha
    return np.mean(SAA_cost), np.var(SAA_cost)

def portfolioObj(x):
    alpha = 0.05
    xval = x[:5]
    cval = x[5]
    mu = -portMu.dot(xval)
    SigmaRoot = np.sqrt(max(xval.dot(portSigma.dot(xval)), 0))
    if SigmaRoot <= 0:
        return cval + max(mu - cval, 0) / alpha
    cutoff = (cval - mu) / SigmaRoot
    def func(t):
        return 1 / np.sqrt(2 * np.pi) * np.exp(-t**2 / 2) * (t - cutoff)
    return cval + SigmaRoot * quad(func, cutoff, np.inf)[0] / alpha
    
def portfolioSample(n, rng):
    return rng.multivariate_normal(portMu, portSigma, size=n)

def portfolioSol():
    alpha = 0.05
    q = norm.ppf(1 - alpha)
    def func(t):
        return 1 / np.sqrt(2 * np.pi) * np.exp(-t**2 / 2) * (t - q)
    kappa = q + quad(func, q, np.inf)[0] / alpha
    m = gp.Model()
    m.Params.OutputFlag = 0
    x = m.addVars(5, obj=-portMu)
    y = m.addVar(obj=kappa)
    m.addConstr(gp.quicksum([portMu[i]*x[i] for i in range(len(x))]) >= portB)
    m.addConstr(gp.quicksum(x) == 1)
    qConstr = gp.QuadExpr()
    for i in range(len(x)):
        for j in range(len(x)):
            qConstr += portSigma[i][j] * x[i] * x[j]
    m.addConstr(qConstr <= y*y)
    m.optimize()
    xval = np.array(m.x[:5])
    mu = -portMu.dot(xval)
    Sigma = max(xval.dot(portSigma.dot(xval)), 0)
    return np.concatenate((xval, [mu + q * np.sqrt(Sigma)]))




# linear integer problem
intSigmaRoot = rng.uniform(size=(10, 10))
intSigma = np.matmul(intSigmaRoot.T, intSigmaRoot)
intMu = np.linspace(-1, 1, num=10)
intB = np.array([-1, 2])
intA1 = -np.ones(10, dtype=int)
intA2 = np.zeros(10, dtype=int)
intA2[-4:] = 1
intA = np.vstack((intA1, intA2))
intXgrid = []
intXval = np.zeros(10, dtype=int)
while True:
    if all(intA.dot(intXval) <= intB):
        intXgrid.append(intXval.copy())
    idx = -1
    intXval[idx] += 1
    while idx > -len(intXval) and intXval[idx] > 1:
        intXval[idx] = 0
        idx -= 1
        intXval[idx] += 1
    if intXval[idx] > 1:
        break
intXgrid = np.array(intXgrid).T
def intSAA(data, x0=None):
    mu = np.mean(data, axis=0)
    idx = np.argmin(mu.dot(intXgrid))
    xval = intXgrid[:, idx]
    SAA_cost = data.dot(xval)
    if x0 is not None:
        SAA_cost -= data.dot(x0)
    return xval, np.mean(SAA_cost), np.var(SAA_cost) # SAA solution, SAA objective value, SAA objective variance

def intSAAObj(data, x, x0=None):
    SAA_cost = data.dot(x)
    if x0 is not None:
        SAA_cost -= data.dot(x0)
    return np.mean(SAA_cost), np.var(SAA_cost)
    
def intObj(x):
    return intMu.dot(x)

def intSample(n, rng):
    return rng.multivariate_normal(intMu, intSigma, size=n)

def intSol():
    idx = np.argmin(intMu.dot(intXgrid))
    return intXgrid[:, idx]
    


linearPerturb = 0.05
# simple linear problem
def linearSAA(data, x0=None):
    mu = np.mean(data)
    slope = -linearPerturb - 2 * mu
    if slope >= 0:
        x = -1
    else:
        x = 1
    SAA_cost = -linearPerturb * x + (3 - 2 * x) * data
    if x0 is not None:
        SAA_cost -= -linearPerturb * x0 + (3 - 2 * x0) * data
    return x, np.mean(SAA_cost), np.var(SAA_cost) # SAA solution, SAA objective value, SAA objective variance

def linearSAAObj(data, x, x0=None):
    SAA_cost = -linearPerturb* x + (3 - 2 * x) * data
    if x0 is not None:
        SAA_cost -= -linearPerturb * x0 + (3 - 2 * x0) * data
    return np.mean(SAA_cost), np.var(SAA_cost)
    
def linearObj(x):
    return -linearPerturb * x

def linearSample(n, rng):
    return rng.normal(size=n)

def linearSol():
    return 1
    
    
    
    