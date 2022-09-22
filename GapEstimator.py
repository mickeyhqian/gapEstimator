#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:24:16 2022

@author: mickey
"""

import numpy as np
from scipy.stats import norm, t


class BagProcedure:
    def __init__(self, replace=False, debias=False, pointVar=False, rng=None):
        self._replace = replace
        self._debias = debias
        self._pointVar = pointVar
        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng
        
    def run(self, GapProblem, data, alpha, k, B, x0=None):
        n = len(data)        
        if n < 30:
            cv = t.ppf(1 - alpha, n - 1)
        else:
            cv = norm.ppf(1 - alpha)
        Ntable = np.full((n, B), -k/n)
        
        Zval = np.zeros(B)
        for b in range(B):
            idxArray = self._rng.choice(n, size=k, replace=self._replace)
            for idx in idxArray:
                Ntable[idx, b] += 1
            _, obj, _ = GapProblem.computeSAA(data[idxArray], x0=x0)
            Zval[b] = obj
    
        Zmean = np.mean(Zval)
        Zvar = np.var(Zval)
        cov = Ntable.dot(Zval - Zmean) / B
        sigma2 = np.sum(cov**2)
        
        if self._debias:
            if self._replace:
                sigma2 = max(sigma2 - k*(n-1) / (B*n) * Zvar, 0)
            else:
                sigma2 = max(sigma2 - k*(n-k) / (B*n) * Zvar, 0)
            
        if not self._replace:
            sigma2 *= n**2 / (n-k)**2
            
        if self._pointVar:
            sigma2 += Zvar / B
        
        if x0 is None:
            return Zmean - cv * np.sqrt(sigma2), Zmean, sigma2
        else:
            return -(Zmean - cv * np.sqrt(sigma2)), -Zmean, sigma2
    
    
class BatchProcedure:
    def run(self, GapProblem, data, alpha, m, x0=None):
        n = len(data)
        k = n // m
        surplus = n - m * k
        sizeList = [k]*(m - surplus) + [k+1]*surplus
        idxArray = np.concatenate(([0], np.cumsum(sizeList)))
        if m < 30:
            cv = t.ppf(1 - alpha, m - 1)
        else:
            cv = norm.ppf(1 - alpha)
        Zval = np.zeros(m)
        for b in range(m):
            _, obj, _ = GapProblem.computeSAA(data[idxArray[b]:idxArray[b+1]], x0=x0)
            Zval[b] = obj
        
        Zmean = np.mean(Zval)
        Zvar = np.var(Zval, ddof=1)
        sigma2 = Zvar / m
            
        if x0 is None:
            return Zmean - cv * np.sqrt(sigma2), Zmean, sigma2
        else:
            return -(Zmean - cv * np.sqrt(sigma2)), -Zmean, sigma2
    
    
class SRPProcedure:        
    def run(self, GapProblem, data, alpha, x0=None):
        n = len(data)
        if n < 30:
            cv = t.ppf(1 - alpha, n - 1)
        else:
            cv = norm.ppf(1 - alpha)

        _, obj, var = GapProblem.computeSAA(data, x0=x0)

        sigma2 = var / n
        
        if x0 is None:
            return obj - cv * np.sqrt(sigma2), obj, sigma2
        else:
            return -(obj - cv * np.sqrt(sigma2)), -obj, sigma2
    
    
class I2RPProcedure:        
    def run(self, GapProblem, data, alpha, x0=None):
        n = len(data)
        n1 = n // 2
        if n1 < 30:
            cv = t.ppf(1 - alpha, n1 - 1)
        else:
            cv = norm.ppf(1 - alpha)

        _, obj, _ = GapProblem.computeSAA(data[:n1], x0=x0)
        _, _, var = GapProblem.computeSAA(data[n1:], x0=x0)

        sigma2 = var / n1
        
        if x0 is None:
            return obj - cv * np.sqrt(sigma2), obj, sigma2
        else:
            return -(obj - cv * np.sqrt(sigma2)), -obj, sigma2
    
    
class A2RPProcedure:        
    def run(self, GapProblem, data, alpha, x0=None):
        n = len(data)
        n1 = n // 2
        if n < 30:
            cv = t.ppf(1 - alpha, n - 1)
        else:
            cv = norm.ppf(1 - alpha)

        _, obj1, var1 = GapProblem.computeSAA(data[:n1], x0=x0)
        _, obj2, var2 = GapProblem.computeSAA(data[n1:], x0=x0)

        obj = (obj1 + obj2) / 2
        var = (var1 + var2) / 2
        sigma2 = var / n
            
        if x0 is None:
            return obj - cv * np.sqrt(sigma2), obj, sigma2
        else:
            return -(obj - cv * np.sqrt(sigma2)), -obj, sigma2
            
    
class GapProblem:
    def __init__(self, SAA, sampleObj):
        self._SAA = SAA
        self._sampleObj = sampleObj
    
    def computeSAA(self, data, x0=None):
        return self._SAA(data, x0=x0)
    
    def getSAAObj(self, data, x, x0=None):
        return self._sampleObj(data, x, x0=x0)