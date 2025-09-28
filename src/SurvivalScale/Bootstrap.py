from multiprocessing import Pool
import numpy as np
import pandas as pd
from tqdm import tqdm
from .Solver import solver


def bootstrap_iteration(tup):
    block_id, columns, pdData = tup
    if block_id is None:
        sample_data = pdData.sample(n=len(pdData), replace=True)
    else:
        unique_blocks = pdData[block_id].unique()
        sampled_blocks = np.random.choice(unique_blocks, size=len(unique_blocks), replace=True)
        sample_data = pd.concat([pdData[pdData[block_id] == block] for block in sampled_blocks])
    sample_data = sample_data.reset_index(drop=True)
    cols = [col for col in columns if sample_data[col].sum() != 0]
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
