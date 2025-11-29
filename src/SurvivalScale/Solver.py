import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, xlogy # pylint: disable = no-name-in-module
from numba import njit


@njit
def _oneRow(params, row_data, cjLength):
    row_identifier = int(row_data[1])
    cj = params[row_identifier]
    row_dataX = row_data[3:].astype(np.float64)
    betaParams = params[cjLength:]
    projection = np.dot(row_dataX, betaParams)
    return [projection + cj, projection, row_data[0], row_data[2]]

@njit
def _oneExpitRow(row_data):
    firstPos = row_data[2] if row_data[0] == 0 else 1
    secondPos = 1 if row_data[0] == 0 else 1 - row_data[2]
    thirdPos = 1 if row_data[0] == 0 else 1 - row_data[3]
    fourthPos = 1 if row_data[0] in (0, int(row_data[1])) else row_data[3]
    return [ firstPos, secondPos, row_data[0] - 1, thirdPos, fourthPos ]

@njit
def _computeExpitRows(cleanedExpitProjections):
    n_rows = cleanedExpitProjections.shape[0]
    formulaPosArray = np.empty((n_rows, 5), dtype=np.float64)
    for i in range(n_rows):
        result = _oneExpitRow(cleanedExpitProjections[i])
        formulaPosArray[i, 0] = result[0]
        formulaPosArray[i, 1] = result[1]
        formulaPosArray[i, 2] = result[2]
        formulaPosArray[i, 3] = result[3]
        formulaPosArray[i, 4] = result[4]
    return formulaPosArray

@njit
def _computeProjections(params, data, cjLength):
    n_rows = data.shape[0]
    projectionsData = np.empty((n_rows, 4), dtype=np.float64)
    for i in range(n_rows):
        result = _oneRow(params, data[i], cjLength)
        projectionsData[i, 0] = result[0]
        projectionsData[i, 1] = result[1]
        projectionsData[i, 2] = result[2]
        projectionsData[i, 3] = result[3]
    return projectionsData

def objective_function(params, data, cjLength):
    projectionsData = _computeProjections(params, data, cjLength)
    projectionsData = np.array(projectionsData, dtype=np.float64)
    cleanedExpitProjections = np.hstack((projectionsData[:, [2, 3]], expit(projectionsData[:, [0, 1]])))
    formulaPosArray = _computeExpitRows(cleanedExpitProjections)
    formulaPosArray = np.hstack((
        xlogy(1, formulaPosArray[:, [0, 1, 4]]),
        xlogy(formulaPosArray[:, 2], formulaPosArray[:, 3]).reshape(-1, 1)
    ))
    return -np.sum(formulaPosArray, dtype=np.float64)

def solver(pdData, columns = None):
    if not columns:
        columns = pdData.columns[3:]
    pdData['question_id'], unique_questions = pdData['question'].factorize()
    cjLength = len(unique_questions)
    betaLength = len(columns)

    numpyArray = np.hstack((
        pdData[['k', 'question_id', 'bound']].to_numpy(),
        pdData[columns].to_numpy()
    ))

    # Initial guess for the parameters
    initial_guess = np.array((1 / (2 * pdData['k'].mean()),) * (cjLength + betaLength), np.dtype(float))
    # Minimize the objective function
    minimum = minimize(
        objective_function,
        initial_guess,
        args=(numpyArray, cjLength),
        method='Powell'
    )
    if not minimum.success:
        raise ValueError('The optimization did not converge')

    solvedParams = minimum.x.flatten().tolist()
    solvedCj = solvedParams[:cjLength]
    cjDataframe = pdData[['question_id', 'question']].drop_duplicates().copy()
    cjDataframe.loc[:,'Cj'] = cjDataframe['question_id'].apply(lambda qid: solvedCj[qid])
    cjDataframe.set_index('question', inplace=True)
    solvedBeta = solvedParams[cjLength:]
    betaDataframe = pd.DataFrame(solvedBeta, columns=['beta'], index=columns)
    return cjDataframe['Cj'], betaDataframe, -minimum.fun, len(solvedParams)

def restricted_solver(pdData):
    data = pdData[['k', 'question', 'bound']].copy()
    data['question_id'], unique_questions = data['question'].factorize()
    data['ones'] = 1

    cjLength = len(unique_questions)
    initial_guess = np.array((1 / (2 * pdData['k'].mean()),) * (cjLength + 1), np.dtype(float))
    
    numpyArray = np.hstack((
        data[['k', 'question_id', 'bound']].to_numpy(),
        data[['ones']].to_numpy()
    ))
    
    minimum = minimize(
        objective_function,
        initial_guess,
        args=(numpyArray, cjLength),
        method='Powell'
    )
    if not minimum.success:
        raise ValueError('The optimization did not converge')
    return -minimum.fun
