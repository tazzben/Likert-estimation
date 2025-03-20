import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, xlogy # pylint: disable = no-name-in-module
from numba import njit


@njit
def _oneRow(params, row_data, cjLength):
    row_identifier = row_data[1]
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

def objective_function(params, data, cjLength):
    projectionsData = np.apply_along_axis(lambda row_data: _oneRow(params, row_data, cjLength), 1, data)
    cleanedExpitProjections = np.hstack((projectionsData[:, [2, 3]], expit(projectionsData[:, [0, 1]])))
    formulaPosArray = np.apply_along_axis(_oneExpitRow, 1, cleanedExpitProjections)
    formulaPosArray = np.hstack((
        xlogy(1, formulaPosArray[:, [0, 1]]),
        xlogy(formulaPosArray[:, 2], formulaPosArray[:, 3]).reshape(-1, 1),
        xlogy(1, formulaPosArray[:, 4]).reshape(-1, 1)
    ))
    return -np.sum(formulaPosArray)

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
    cjDataframe = pdData[['question_id', 'question']].drop_duplicates()
    cjDataframe['Cj'] = cjDataframe['question_id'].apply(lambda qid: solvedCj[qid])
    cjDataframe.set_index('question', inplace=True)
    solvedBeta = solvedParams[cjLength:]
    betaDataframe = pd.DataFrame(solvedBeta, columns=['beta'], index=columns)
    return cjDataframe['Cj'], betaDataframe


def testFunction():
    data = pd.DataFrame({
        'k': [0, 1, 2, 3, 4, 0, 1, 2, 8, 9],
        'question': ['q1', 'q1', 'q1', 'q1', 'q1', 'q2', 'q2', 'q2', 'q2', 'q2'],
        'bound': [4, 4, 4, 4, 4, 9, 9, 9, 9, 9],
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    })
    cj, beta = solver(data, columns=['feature1', 'feature2'])
    print(cj)
    print(beta)

if __name__ == '__main__':
    testFunction()
