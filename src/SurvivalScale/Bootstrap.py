from multiprocessing import Pool
import numpy as np
import pandas as pd
from tqdm import tqdm
from .Solver import solver
from scipy.special import expit # pylint: disable = no-name-in-module
from numba import njit


def bootstrap_iteration(tup):
    block_id, columns, pdData = tup
    if block_id is None:
        sample_data = pdData.sample(n=len(pdData), replace=True)
    else:
        unique_blocks = pdData[block_id].unique()
        sampled_blocks = np.random.choice(unique_blocks, size=len(unique_blocks), replace=True)
        sample_data = pd.concat([pdData[pdData[block_id] == block] for block in sampled_blocks])
    sample_data = sample_data.reset_index(drop=True)
    cols = [col for col in columns if (sample_data[col].sum() != 0 or sample_data[col].nunique() > 1)]
    cjResults, solvedParams, _, _ = solver(sample_data, cols)
    cjResults = cjResults.reset_index()
    solvedParams = solvedParams.reset_index()
    return cjResults, solvedParams

def bootstrap(pdData, columns=None, n_bootstraps=1000, alpha=0.05, block_id=None):
    if not columns:
        columns = pdData.columns[3:]
        columns = list(columns)

    rows = [(block_id, columns, pdData) for _ in range(n_bootstraps)]
    with Pool() as pool:
        results = list(tqdm(pool.imap(bootstrap_iteration, rows), total=n_bootstraps))

    # Combine results from all iterations
    cj_results = pd.concat([result[0] for result in results], axis=0, ignore_index=True)
    beta_results = pd.concat([result[1] for result in results], axis=0, ignore_index=True)

    unique_questions = cj_results['question'].unique()

    cj_list = []
    beta_list = []

    for question in unique_questions:
        question_mask = cj_results['question'] == question
        median_cj = cj_results[question_mask]['Cj'].median()
        lower_bound = cj_results[question_mask]['Cj'].quantile(alpha / 2)
        upper_bound = cj_results[question_mask]['Cj'].quantile(1 - alpha / 2)
        p_value = ((cj_results[question_mask]['Cj'] < 0).sum()) / (len(cj_results[question_mask]['Cj'])) if median_cj > 0 else ((cj_results[question_mask]['Cj'] > 0).sum()) / (len(cj_results[question_mask]['Cj']))
        cj_list.append({
            'question': question,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'p_value': p_value * 2
        })

    for col in columns:
        col_mask = beta_results['index'] == col
        median_beta = beta_results[col_mask]['beta'].median()
        lower_bound = beta_results[col_mask]['beta'].quantile(alpha / 2)
        upper_bound = beta_results[col_mask]['beta'].quantile(1 - alpha / 2)
        p_value = ((beta_results[col_mask]['beta'] < 0).sum()) / (len(beta_results[col_mask]['beta'])) if median_beta > 0 else ((beta_results[col_mask]['beta'] > 0).sum()) / (len(beta_results[col_mask]['beta']))
        beta_list.append({
            'variable': col,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'p_value': p_value * 2
        })
    cj_results = pd.DataFrame(cj_list)
    beta_results = pd.DataFrame(beta_list)
    return cj_results, beta_results


@njit
def findFailureBin(cjprob, nocjprob, bound, rng):
    if rng.random() < cjprob:
        return 0
    for bin in range(1, bound + 1):   
        if rng.random() < nocjprob:
            return bin
    return bound

def parametric_bootstrap_iteration(tup):
    columns, pdData = tup
    rng = np.random.default_rng()
    ystars = [findFailureBin(row['expit_projection_cj'], row['expit_projection_nocj'], int(row['bound']), rng) for _, row in pdData.iterrows()]
    pdData = pdData.copy()
    pdData['k'] = ystars
    cjResults, solvedParams, _, _ = solver(pdData, columns)
    cjResults = cjResults.reset_index()
    solvedParams = solvedParams.reset_index()
    return cjResults, solvedParams
 
def parametric_bootstrap_correction(pdData, betasO, cjsO, columns=None, n_bootstraps=1000):
    pbData = pdData.copy()
    betas = betasO.copy()
    cjs = cjsO.copy()
    if not columns:
        columns = pbData.columns[3:]
        columns = list(columns)
    betas_numpy = betas.loc[columns,'beta'].to_numpy()
    xrows = pbData[columns].to_numpy()
    projection_nocj = np.dot(xrows, betas_numpy)
    pbData['projection_nocj'] = projection_nocj
    cjs.index.name = 'question'
    cjsdata = cjs.reset_index()
    pbData = pbData.merge(cjsdata, on='question', how='left')
    pbData['projection_cj'] = pbData['projection_nocj'] + pbData['Cj']
    pbData['expit_projection_cj'] = expit(pbData['projection_cj'])
    pbData['expit_projection_nocj'] = expit(pbData['projection_nocj'])

    rows = [(columns, pbData) for _ in range(n_bootstraps)]

    with Pool() as pool:
        results = list(tqdm(pool.imap(parametric_bootstrap_iteration, rows), total=n_bootstraps))

    # Combine results from all iterations
    cj_results = pd.concat([result[0] for result in results], axis=0, ignore_index=True)
    beta_results = pd.concat([result[1] for result in results], axis=0, ignore_index=True)

    unique_questions = cj_results['question'].unique()

    cj_list = []
    beta_list = []

    for question in unique_questions:
        question_mask = cj_results['question'] == question
        cjStar = cj_results[question_mask]['Cj'].mean()
        bias = cjStar - cjs.loc[question]
        cj_list.append({
            'question': question,
            'bias': bias,
            'cJ': cjStar,
            'corrected_Cj': cjs.loc[question] - bias
        })

    for col in columns:
        col_mask = beta_results['index'] == col
        betaStar = beta_results[col_mask]['beta'].mean()
        bias = betaStar - betas.loc[col, 'beta']
        beta_list.append({
            'variable': col,
            'bias': bias,
            'beta': betaStar,
            'corrected_beta': betas.loc[col, 'beta'] - bias
        })
    
    cj_results = pd.DataFrame(cj_list)
    beta_results = pd.DataFrame(beta_list)
    return cj_results, beta_results
    