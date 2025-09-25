import numpy as np
import pandas as pd
from scipy.special import expit # pylint: disable = no-name-in-module

def marginal_func(beta, xrow, beta_position, adder = 0, multiplierOverride = None):
    projection = np.dot(beta, xrow) + adder
    if multiplierOverride is not None:
        multiplier = multiplierOverride
    else:
        multiplier = beta[beta_position]
    marginal = multiplier * expit(projection) * (1 - expit(projection))
    return marginal

def discrete_marginal_func(beta, xrow, beta_position, binary = False, adder = 0, adderBase = 0, overrideBetaModifier = None):
    newXrow = xrow.copy()
    oneXrow = xrow.copy()
    if overrideBetaModifier is None:
        newXrow[beta_position] = 0
        oneXrow[beta_position] = 1
        binary = True        
    projection0 = np.dot(beta, newXrow) + adderBase
    projection1 = np.dot(beta, oneXrow) + adder
    div = 1 if binary else xrow[beta_position]
    if div == 0:
        return np.nan
    marginal = (expit(projection1) - expit(projection0)) / div
    return marginal

def marginalize(xrow, beta, beta_position, discrete=False):
    if discrete:
        testColumn = xrow[:, beta_position]
        binary = True if np.all(np.isin(testColumn, [0, 1])) else False
        return np.array([ discrete_marginal_func(beta, xrow[i], beta_position, binary) for i in range(len(xrow)) ]).mean()
    return np.array([ marginal_func(beta, xrow[i], beta_position) for i in range(len(xrow)) ]).mean()

def marginalize_cj(xrow, beta, cjValue, discrete=False):
    if discrete:
        return np.array([ discrete_marginal_func(beta, xrow[i], 0, adder=cjValue, overrideBetaModifier=True, binary=True) for i in range(len(xrow)) ]).mean()
    return np.array([ marginal_func(beta, xrow[i], 0, adder=cjValue, multiplierOverride=cjValue) for i in range(len(xrow)) ]).mean()

def marginalize_df(df, beta, columns, discrete=False):
    data = df[columns].to_numpy()
    estimatedBeta = beta.loc[columns].to_numpy().flatten()
    m = {
        'marginal': [ marginalize(data, estimatedBeta, i, discrete) for i in range(len(columns)) ], 
        'variable': columns
    }
    return pd.DataFrame(m).set_index('variable')

def marginalize_cjdf(df, beta, cjdf, columns, discrete=False):
    data = df[columns].to_numpy()
    estimatedBeta = beta.loc[columns].to_numpy().flatten()
    m = {
        'marginal': [ marginalize_cj(data, estimatedBeta, cjdf.loc[var], discrete) for var in cjdf.keys() ],
        'variable': cjdf.keys()
    }
    return pd.DataFrame(m).set_index('variable')
