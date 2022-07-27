#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 18:29:36 2022

@author: mickey
"""

import numpy as np
from scipy.stats import norm
from GapEstimator import BagProcedure, BatchProcedure
import logging
logger = logging.getLogger("gapEstimator")


def computeOptVal(gapProblem, gapEstimator, computeObjVal, getOptSol, getSample, alpha, n, numTrial, m=None, k=None, B=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    trueSol = getOptSol()
    trueObj = computeObjVal(trueSol)
    bound = np.zeros(numTrial)
    for i in range(numTrial):
        # generate sample
        data = getSample(n, rng)
        
        # compute lower bound of optimal value
        if isinstance(gapEstimator, BagProcedure):
            assert k is not None
            assert B is not None
            newBound, _, _ = gapEstimator.run(gapProblem, data, alpha, k, B)
        elif isinstance(gapEstimator, BatchProcedure):
            assert m is not None
            newBound, _, _ = gapEstimator.run(gapProblem, data, alpha, m)
        else:
            newBound, _, _ = gapEstimator.run(gapProblem, data, alpha)
            
        bound[i] = newBound
        if (i+1) % 50 == 0:
            logger.info(f"{type(gapEstimator).__name__}: Finish trial # {i+1}")
        
    return np.mean(bound <= trueObj), np.mean(bound), np.var(bound)


def computeGapBC(gapProblem, gapEstimator, computeObjVal, getOptSol, getSample, alpha, n1, n2, numTrial, m=None, k=None, B=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    trueSol = getOptSol()
    trueObj = computeObjVal(trueSol)
    bound = np.zeros(numTrial)
    gap = np.zeros(numTrial)
    n = n1 + n2
    for i in range(numTrial):
        # generate sample
        data = getSample(n, rng)
        x0, _, _ = gapProblem.computeSAA(data[:n1])
        gap[i] = computeObjVal(x0) - trueObj
        mean, var = gapProblem.getSAAObj(data[n1:], x0)
        upper = mean + norm.ppf(1 - alpha/2) * np.sqrt(var / n2)
        
        # compute lower bound of optimal value
        if isinstance(gapEstimator, BagProcedure):
            assert k is not None
            assert B is not None
            lower, _, _ = gapEstimator.run(gapProblem, data, alpha/2, k, B)
        elif isinstance(gapEstimator, BatchProcedure):
            assert m is not None
            lower, _, _ = gapEstimator.run(gapProblem, data, alpha/2, m)
        else:
            lower, _, _ = gapEstimator.run(gapProblem, data, alpha/2)
            
        bound[i] = max(upper - lower, 0)
        if (i+1) % 50 == 0:
            logger.info(f"{type(gapEstimator).__name__}: Finish trial # {i+1}")
        
    return np.mean(bound >= gap), np.mean(bound), np.var(bound)


def computeGapCRN(gapProblem, gapEstimator, computeObjVal, getOptSol, getSample, alpha, n1, n2, numTrial, m=None, k=None, B=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    trueSol = getOptSol()
    trueObj = computeObjVal(trueSol)
    bound = np.zeros(numTrial)
    gap = np.zeros(numTrial)
    n = n1 + n2
    for i in range(numTrial):
        # generate sample
        data = getSample(n, rng)
        x0, _, _ = gapProblem.computeSAA(data[:n1])
        gap[i] = computeObjVal(x0) - trueObj
        # compute lower bound of optimal value
        if isinstance(gapEstimator, BagProcedure):
            assert k is not None
            assert B is not None
            lower, _, _ = gapEstimator.run(gapProblem, data[n1:], alpha, k, B, x0=x0)
        elif isinstance(gapEstimator, BatchProcedure):
            assert m is not None
            lower, _, _ = gapEstimator.run(gapProblem, data[n1:], alpha, m, x0=x0)
        else:
            lower, _, _ = gapEstimator.run(gapProblem, data[n1:], alpha, x0=x0)
            
        bound[i] = max(-lower, 0)
        if (i+1) % 50 == 0:
            logger.info(f"{type(gapEstimator).__name__}: Finish trial # {i+1}")
        
    return np.mean(bound >= gap), np.mean(bound), np.var(bound)