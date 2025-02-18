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
def oneRow(params, row_data, cjLength):
    row_identifier = row_data[2]
    cj = params[row_identifier]
    row_dataX = row_data[3:]
    betaParams = params[cjLength:]
    projection = np.dot(row_dataX, betaParams)
    return [projection + cj, projection, row_data[0], row_data[1]]

@njit
def oneExpitRow(row_data):
    firstPos = row_data[2] if row_data[0] == 0 else 1
    secondPos = 1 if row_data[0] == 0 else 1 - row_data[2]
    thirdPos = 1 if row_data[0] == 0 else 1 - row_data[3]
    fourthPos = 1 if (row_data[0] == row_data[1] or row_data[0] == 0) else row_data[3]
    return [ firstPos, secondPos, row_data[0] - 1, thirdPos, fourthPos ]

def objective_function(params, data, cjLength):
    projectionsData = np.apply_along_axis(lambda row_data: oneRow(params, row_data, cjLength), 1, data)
    projectionsData[:,4] = expit(projectionsData[:,0])
    projectionsData[:,5] = expit(projectionsData[:,1])
    cleanedExpitProjections = projectionsData[:,[2, 3, 4, 5]]
    formulaPosArray = np.array([ oneExpitRow(row_data) for row_data in cleanedExpitProjections ])
    formulaPosArray[:,5] = np.log(formulaPosArray[:,0])
    formulaPosArray[:,6] = np.log(formulaPosArray[:,1])
    formulaPosArray[:,7] = xlogy(formulaPosArray[:,2], formulaPosArray[:,3])
    formulaPosArray[:,8] = np.log(formulaPosArray[:,4])
    return -np.sum(formulaPosArray[:,5] + formulaPosArray[:,6] + formulaPosArray[:,7] + formulaPosArray[:,8])
    