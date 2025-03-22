import numpy as np
from scipy.stats.distributions import chi2
from .Solver import solver, restricted_solver
from .Marginal import marginalize_df
from .Bootstrap import bootstrap

def get_results(data, columns=None, bootstrap_iterations=1000, alpha=0.05, block_id=None):
    """
    Get the results of the SurvivalScale analysis.

    Parameters:
    - data (pd.DataFrame): The input data containing the features and other necessary columns.
    - columns (list, optional): The list of feature columns to be used in the analysis. 
                                 If None, all columns except 'k', 'question', 'bound' will be used.

    Returns:
    - pd.DataFrame: question-level results (Cj values).
    - pd.DataFrame: variable-level results (beta values and their marginals).
    - dict: metrics including McFadden pseudo R-squared, log-likelihood, AIC, BIC, etc.
    """
    if columns is None:
        columns = [col for col in data.columns if col not in ['k', 'question', 'bound']]
    # Solve the model
    cj, beta, fun, parlen = solver(data, columns=columns)
    # Marginalize the results
    marginals = marginalize_df(data, beta, columns=columns, discrete=False)
    marginalD = marginalize_df(data, beta, columns=columns, discrete=True)
    # Merge beta and marginals on key
    results = beta.join(marginals, how='inner')
    marginalD = marginalD.rename(columns={'marginal': 'marginal_discrete'})
    results = results.join(marginalD, how='inner')
    # Perform bootstrap analysis
    cjbootstrap, bootstrap_results = bootstrap(data, n_bootstraps=bootstrap_iterations, alpha=alpha, columns=columns, block_id=block_id)
    cjbootstrap = cjbootstrap.set_index('question')
    bootstrap_results = bootstrap_results.set_index('variable')
    results = results.join(bootstrap_results, how='inner')
    cj = cj.to_frame().join(cjbootstrap, how='inner')


    # Calculate McFadden pseudo R-squared
    restrictedFun = restricted_solver(data)

    metrics = {
        'McFadden_R2': 1 - (fun / restrictedFun),
        'log_likelihood': fun,
        'restricted_log_likelihood': restrictedFun,
        'LR': -2 * (restrictedFun - fun)
    }

    metrics['num_rows'] = data.shape[0]
    metrics['Chi-Squared_p-value'] = chi2.sf(metrics['LR'], (parlen - 1))
    metrics['AIC'] = 2*(parlen)-2*fun
    metrics['BIC'] = (parlen) * np.log(metrics['num_rows'])-2*fun

    return cj, results, metrics