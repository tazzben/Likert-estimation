from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import percentileofscore
from scipy.stats.distributions import chi2
from scipy.special import expit, xlog1py, xlogy
from tqdm import tqdm
from numba import njit


@njit
def _oneRow(params, row_data, cjLength):
    row_identifier = row_data[2]
    cj = params[row_identifier]
    row_dataX = row_data[3:]
    betaParams = params[cjLength:]
    projection = np.dot(row_dataX, betaParams)
    return [projection + cj, projection, row_data[0], row_data[1]]

@njit
def _oneExpitRow(row_data):
    firstPos = row_data[2] if row_data[0] == 0 else 1
    secondPos = 1 if row_data[0] == 0 else 1 - row_data[2]
    thirdPos = 1 if row_data[0] == 0 else 1 - row_data[3]
    fourthPos = 1 if row_data[0] in (0, row_data[1]) else row_data[3]
    return [ firstPos, secondPos, row_data[0] - 1, thirdPos, fourthPos ]

def objective_function(params, data, cjLength):
    projectionsData = np.apply_along_axis(lambda row_data: _oneRow(params, row_data, cjLength), 1, data)
    
    cleanedExpitProjections = np.hstack((projectionsData[:, [2, 3]], expit(projectionsData[:, [0, 1]])))
        
    formulaPosArray = np.apply_along_axis(lambda row_data: _oneExpitRow(row_data), 1, cleanedExpitProjections)
    
    formulaPosArray = np.hstack((
        np.log(formulaPosArray[:, [0, 1]]),
        xlogy(formulaPosArray[:, 2], formulaPosArray[:, 3]).reshape(-1, 1),
        np.log(formulaPosArray[:, 4]).reshape(-1, 1)
    ))
    
    return -np.sum(formulaPosArray)

def solver(pdData):
    