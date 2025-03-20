import numpy as np
from scipy.special import expit # pylint: disable = no-name-in-module

def marginal_func(beta, xrow, beta_position):
    projection = np.dot(beta, xrow)
    marginal = beta[beta_position]*1/(2+2*np.cosh(projection))
    return marginal

def discrete_marginal_func(beta, xrow, beta_position):
    newBeta = beta.copy()
    newBeta[beta_position] = 0
    projection0 = np.dot(newBeta, xrow)
    projection1 = np.dot(beta, xrow)
    marginal = (expit(projection1) - expit(projection0))
    return marginal

