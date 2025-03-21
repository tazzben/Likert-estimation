import numpy as np
import pandas as pd
from scipy.special import expit # pylint: disable = no-name-in-module

def marginal_func(beta, xrow, beta_position):
    projection = np.dot(beta, xrow)
    marginal = beta[beta_position] * expit(projection) * (1 - expit(projection))
    return marginal

def discrete_marginal_func(beta, xrow, beta_position):
    newBeta = beta.copy()
    newBeta[beta_position] = 0
    projection0 = np.dot(newBeta, xrow)
    projection1 = np.dot(beta, xrow)
    marginal = (expit(projection1) - expit(projection0))
    return marginal

def marginalize(xrow, beta, beta_position, discrete=False):
    if discrete:
        return np.array([discrete_marginal_func(beta, xrow[i], beta_position) for i in range(len(xrow))]).mean()
    return np.array([marginal_func(beta, xrow[i], beta_position) for i in range(len(xrow))]).mean()

def marginalize_df(df, beta, columns, discrete=False):
    data = df[columns].to_numpy()
    estimatedBeta = beta.loc[columns].to_numpy().flatten()
    m = {
        'marginal': [ marginalize(data, estimatedBeta, i, discrete) for i in range(len(columns)) ], 
        'variable': columns
    }
    return pd.DataFrame(m).set_index('variable')

