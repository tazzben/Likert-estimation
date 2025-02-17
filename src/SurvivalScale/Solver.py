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
