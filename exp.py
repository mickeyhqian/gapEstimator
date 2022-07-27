#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 21:54:31 2022

@author: mickey
"""

import gurobipy as gp
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from Example import cvarSAA, cvarObj, cvarSAAObj, cvarSample, cvarSol, portfolioSAA, portfolioObj, portfolioSAAObj, portfolioSample, portfolioSol, intSAA, intObj, intSAAObj, intSample, intSol, linearSAA, linearObj, linearSAAObj, linearSample, linearSol
from GapEstimator import BagProcedure, BatchProcedure, SRPProcedure, I2RPProcedure, A2RPProcedure, GapProblem
from ExpProcedure import computeOptVal, computeGapBC, computeGapCRN
import sys
import os
import logging
logger = logging.getLogger("gapEstimator")

def main(problem, goal, method, n, n1=None, n2=None, m=None, k=None, B=None, reuseLog=False):
    directory = "/Users/mickey/Documents/ResearchProject/OptimalityGap/"
    file_name = [problem, goal, method, str(n), str(n1), str(n2), str(m), str(k), str(B)]
    file_name = "_".join(file_name)
    file_path = directory + file_name
    if not reuseLog:
        file_path_to_test = file_path
        idx = 0
        while os.path.isfile(file_path_to_test):
            file_path_to_test = file_path + "_" + str(idx)
            idx += 1
        file_path = file_path_to_test
    
    
    handler = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.info("Experiment starts ==========================")
    logger.info(f"problem: {problem}")
    logger.info(f"goal: {goal}")
    logger.info(f"method: {method}")
    logger.info(f"n: {n}")
    logger.info(f"n1: {n1}")
    logger.info(f"n2: {n2}")
    logger.info(f"m: {m}")
    logger.info(f"k: {k}")
    logger.info(f"B: {B}")
    
    
    if problem == "cvar":
        gapProblem = GapProblem(cvarSAA, cvarSAAObj)
        objFunc = cvarObj
        solFunc = cvarSol
        sampleFunc = cvarSample
    elif problem == "portfolio":
        gapProblem = GapProblem(portfolioSAA, portfolioSAAObj)
        objFunc = portfolioObj
        solFunc = portfolioSol
        sampleFunc = portfolioSample
    elif problem == "integer":
        gapProblem = GapProblem(intSAA, intSAAObj)
        objFunc = intObj
        solFunc = intSol
        sampleFunc = intSample
    elif problem == "linear":
        gapProblem = GapProblem(linearSAA, linearSAAObj)
        objFunc = linearObj
        solFunc = linearSol
        sampleFunc = linearSample
    else:
        logger.info("Invalid problem")
        return
    
    
    if method == "SRP":
        gapEstimator = SRPProcedure()
    elif method == "I2RP":
        gapEstimator = I2RPProcedure()
    elif method == "A2RP":
        gapEstimator = A2RPProcedure()
    elif method == "Batch":
        gapEstimator = BatchProcedure()
        assert m is not None
    elif method == "BagU":
        resample_rng = np.random.default_rng(seed=666)
        gapEstimator = BagProcedure(replace=False, debias=True, pointVar=True, rng=resample_rng)
        assert k is not None
        assert B is not None
    elif method == "BagV":
        resample_rng = np.random.default_rng(seed=666)
        gapEstimator = BagProcedure(replace=True, debias=True, pointVar=True, rng=resample_rng)
        assert k is not None
        assert B is not None
    else:
        logger.info("Invalid method")
        return
    
    alpha = 0.05
    logger.info(f"alpha: {alpha}")
    numTrial = 1000
    logger.info(f"#repetition: {numTrial}")
    data_rng = np.random.default_rng(seed=123)
    
    if goal == "OptVal":
        if method == "Batch":
            logger.info(f"Batch size: {n//m}")
        result = computeOptVal(gapProblem, gapEstimator, objFunc, solFunc, sampleFunc, alpha, n, numTrial, m=m, k=k, B=B, rng=data_rng)
    elif goal == "GapBC":
        assert n1 is not None
        assert n2 is not None
        assert n == n1+n2
        if method == "Batch":
            logger.info(f"Batch size: {n//m}")
        result = computeGapBC(gapProblem, gapEstimator, objFunc, solFunc, sampleFunc, alpha, n1, n2, numTrial, m=m, k=k, B=B, rng=data_rng)
    elif goal == "GapCRN":
        assert n1 is not None
        assert n2 is not None
        assert n == n1+n2
        if method == "Batch":
            logger.info(f"Batch size: {n2//m}")
        result = computeGapCRN(gapProblem, gapEstimator, objFunc, solFunc, sampleFunc, alpha, n1, n2, numTrial, m=m, k=k, B=B, rng=data_rng)
    else:
        logger.info("Invalid goal")
        return
    
    logger.info(f"mean coverage: {result[0]}")
    logger.info(f"mean bound: {result[1]}")
    logger.info(f"std bound: {result[2]}")
    logger.info("Experiment finishes ==========================\n\n\n")
    logger.removeHandler(handler)




if __name__ == "__main__":
    n1 = None
    n2 = None
    m = None
    k = None
    B = None
    
    numArg = len(sys.argv)
    idx = 5
    while idx < numArg:
        key, value = sys.argv[idx].split("=")
        if key == "n1":
            n1 = int(value)
        elif key == "n2":
            n2 = int(value)
        elif key == "m":
            m = int(value)
        elif key == "k":
            k = int(value)
        elif key == "B":
            B = int(value)
        else:
            sys.exit(f"Invalid inputs at position {idx}")
        idx += 1
    for k in np.arange(2, 31, 4):
        main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), n1=n1, n2=n2, m=m, k=k, B=B)







