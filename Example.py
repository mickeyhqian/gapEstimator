#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 22:34:21 2022

@author: mickey
"""

import numpy as np
from scipy.stats import norm
from scipy.integrate import quad, dblquad
import gurobipy as gp
from sklearn.linear_model import LogisticRegression
from scipy.linalg import sqrtm


# CVaR problem
cvarAlpha = 0.1
def cvarSAA(data, x0=None):
    n = len(data)
    data = np.sort(data)
    idx = int(np.ceil(cvarAlpha * n))
    x = data[-idx]
    mean, var = cvarSAAObj(data, x, x0=x0)
    return x, mean, var # SAA solution, SAA objective value, SAA objective variance

def cvarSAAObj(data, x, x0=None):
    SAA_cost = x + np.maximum(data - x, 0) / cvarAlpha
    if x0 is not None:
        SAA_cost -= x0 + np.maximum(data - x0, 0) / cvarAlpha
    return np.mean(SAA_cost), np.var(SAA_cost)
    
def cvarObj(x):
    def func(t):
        return 1 / np.sqrt(2 * np.pi) * np.exp(-t**2 / 2) * (t - x)
    return x + quad(func, x, np.inf)[0] / cvarAlpha

def cvarSample(n, rng):
    return rng.normal(size=n)

def cvarSol():
    return norm.ppf(1 - cvarAlpha)


# Portfolio problem
rng = np.random.default_rng(seed=1235)
portSigmaRoot = rng.uniform(size=(5, 5))
portSigma = np.matmul(portSigmaRoot.T, portSigmaRoot)
portMu = np.arange(1, 6)
portB = 3
portAlpha = 0.05
def portfolioSAA(data, x0=None):
    n = len(data)
    m = gp.Model()
    m.Params.OutputFlag = 0
    x = m.addVars(5)
    c = m.addVar(lb=-float("inf"), obj=1)
    s = m.addVars(n, obj=1/(n*portAlpha))
    m.addConstrs(gp.quicksum([-data[i, j]*x[j] for j in range(len(x))]) - c <= s[i] for i in range(len(s)))
    m.addConstr(gp.quicksum([portMu[i]*x[i] for i in range(len(x))]) >= portB)
    m.addConstr(gp.quicksum(x) == 1)
    m.optimize()
    sol = np.array(m.x[:6])
    mean, var = portfolioSAAObj(data, sol, x0=x0)
    return sol, mean, var

def portfolioSAAObj(data, x, x0=None):
    xval = np.array(x[:5])
    cval = x[5]
    SAA_cost = cval + np.maximum(-data.dot(xval) - cval, 0) / portAlpha
    if x0 is not None:
        SAA_cost -= x0[5] + np.maximum(-data.dot(x0[:5]) - x0[5], 0) / portAlpha
    return np.mean(SAA_cost), np.var(SAA_cost)

def portfolioObj(x):
    xval = x[:5]
    cval = x[5]
    mu = -portMu.dot(xval)
    SigmaRoot = np.sqrt(max(xval.dot(portSigma.dot(xval)), 0))
    if SigmaRoot <= 0:
        return cval + max(mu - cval, 0) / portAlpha
    cutoff = (cval - mu) / SigmaRoot
    def func(t):
        return 1 / np.sqrt(2 * np.pi) * np.exp(-t**2 / 2) * (t - cutoff)
    return cval + SigmaRoot * quad(func, cutoff, np.inf)[0] / portAlpha
    
def portfolioSample(n, rng):
    return rng.multivariate_normal(portMu, portSigma, size=n)

def portfolioSol():
    q = norm.ppf(1 - portAlpha)
    def func(t):
        return 1 / np.sqrt(2 * np.pi) * np.exp(-t**2 / 2) * (t - q)
    kappa = q + quad(func, q, np.inf)[0] / portAlpha
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
    mean, var = intSAAObj(data, xval, x0=x0)
    return xval, mean, var # SAA solution, SAA objective value, SAA objective variance

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
    mean, var = linearSAAObj(data, x, x0=x0)
    return x, mean, var # SAA solution, SAA objective value, SAA objective variance

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
    
    
    
# logistic regression
logitD = 10
logitCoeff = rng.uniform(size=logitD+1)
logitEps = 1e-20
def logitSAA(data, x0=None):
    X = data[:, :-1]
    Y = data[:, -1].astype(int)
    oneClass = all(Y == Y[0])
    if oneClass:
        x = np.zeros(logitD+1)
        if Y[0] == 0:
            x[-1] = -1e20
        else:
            x[-1] = 1e20
    else:
        model = LogisticRegression(penalty="none", random_state=666)
        result = model.fit(X, Y)
        x = np.concatenate((result.coef_[0], result.intercept_))
    mean, var = logitSAAObj(data, x, x0=x0)
    return x, mean, var

def logitSAAObj(data, x, x0=None):
    X = data[:, :-1]
    Y = data[:, -1].astype(int)
    posYIdx = Y == 1
    pExp = -X.dot(x[:-1]) - x[-1]
    p = np.zeros(len(pExp))
    posIdx = pExp > 0
    p[posIdx] = np.exp(-pExp[posIdx]) / (np.exp(-pExp[posIdx]) + 1)
    p[~posIdx] = 1 / (np.exp(pExp[~posIdx]) + 1)
    SAA_cost = np.zeros(len(X))
    SAA_cost[posYIdx] = -np.log(np.maximum(p[posYIdx], logitEps))
    SAA_cost[~posYIdx] = -np.log(np.maximum(1 - p[~posYIdx], logitEps))
    if x0 is not None:
        p0Exp = -X.dot(x0[:-1]) - x0[-1]
        p0 = np.zeros(len(p0Exp))
        posIdx = p0Exp > 0
        p0[posIdx] = np.exp(-p0Exp[posIdx]) / (np.exp(-p0Exp[posIdx]) + 1)
        p0[~posIdx] = 1 / (np.exp(p0Exp[~posIdx]) + 1)
        SAA_cost[posYIdx] -= -np.log(np.maximum(p0[posYIdx], logitEps))
        SAA_cost[~posYIdx] -= -np.log(np.maximum(1 - p0[~posYIdx], logitEps))
    return np.mean(SAA_cost), np.var(SAA_cost)

def logitObj(x):
    A = np.vstack((logitCoeff[:-1], x[:-1]))
    Aroot = sqrtm(A.dot(A.T))
    def func(a, b):
        density = 1 / (2 * np.pi) * np.exp(-(a**2 + b**2) / 2)
        transform = Aroot.dot([a,b])
        pstarExp = -transform[0] - logitCoeff[-1]
        if pstarExp > 0:
            pstar = np.exp(-pstarExp) / (np.exp(-pstarExp) + 1)
        else:
            pstar = 1 / (np.exp(pstarExp) + 1)
        pExp = -transform[1] - x[-1]
        if pExp > 0:
            p = np.exp(-pExp) / (np.exp(-pExp) + 1)
        else:
            p = 1 / (np.exp(pExp) + 1)
        val = -pstar * np.log(np.maximum(p, logitEps)) - (1 - pstar) * np.log(np.maximum(1 - p, logitEps))
        return val * density
    return dblquad(func, -np.inf, np.inf, -np.inf, np.inf)[0]
    
def logitSample(n, rng):
    X = rng.normal(size=(n, logitD))
    Y = []
    for i in range((len(X))):
        pExp = -X[i].dot(logitCoeff[:-1]) - logitCoeff[-1]
        if pExp > 0:
            p = np.exp(-pExp) / (np.exp(-pExp) + 1)
        else:
            p = 1 / (np.exp(pExp) + 1)
        if rng.uniform() < p:
            Y.append(1)
        else:
            Y.append(0)
    return np.hstack((X, np.array(Y).reshape(-1, 1)))

def logitSol():
    return logitCoeff
    
    
    
    

simplexMu = np.zeros(10)
simplexMu[5:] = 0.1
# simple linear problem
def simplexSAA(data, x0=None):
    mu = np.mean(data, axis=0)
    idx = np.argmin(mu)
    x = np.zeros(10)
    x[idx] = 1
    mean, var = simplexSAAObj(data, x, x0=x0)
    return x, mean, var # SAA solution, SAA objective value, SAA objective variance

def simplexSAAObj(data, x, x0=None):
    SAA_cost = data.dot(x)
    if x0 is not None:
        SAA_cost -= data.dot(x0)
    return np.mean(SAA_cost), np.var(SAA_cost)
    
def simplexObj(x):
    return simplexMu.dot(x)

def simplexSample(n, rng):
    return rng.multivariate_normal(simplexMu, np.eye(10), size=n)

def simplexSol():
    idx = np.argmin(simplexMu)
    sol = np.zeros(10)
    sol[idx] = 1
    return sol
    
    
    