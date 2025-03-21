import numpy as np
import pandas as pd
from .Solver import solver
from .Marginal import marginalize_df
from .Bootstrap import bootstrap

def get_results(data, columns=None, bootstrap_iterations=1000, alpha=0.05):
    """
    Get the results of the SurvivalScale analysis.

    Parameters:
    - data (pd.DataFrame): The input data containing the features and other necessary columns.
    - columns (list, optional): The list of feature columns to be used in the analysis. 
                                 If None, all columns except 'k', 'question', 'bound' will be used.

    Returns:
    - pd.DataFrame: A DataFrame containing the results of the analysis.
    """
    
    if columns is None:
        columns = [col for col in data.columns if col not in ['k', 'question', 'bound']]
    
    # Solve the model
    cj, beta, fun = solver(data, columns=columns)
    
    # Marginalize the results
    marginals = marginalize_df(data, beta, columns=columns, discrete=False)
    marginalD = marginalize_df(data, beta, columns=columns, discrete=True)

    # Merge beta and marginals on key
    results = beta.join(marginals, how='inner')
    marginalD = marginalD.rename(columns={'marginal': 'marginal_discrete'})
    results = results.join(marginalD, how='inner')
    
    # Perform bootstrap analysis
    cjbootstrap, bootstrap_results = bootstrap(data, n_bootstraps=bootstrap_iterations, alpha=alpha, columns=columns)
    cjbootstrap = cjbootstrap.set_index('question')
    bootstrap_results = bootstrap_results.set_index('variable')
    results = results.join(bootstrap_results, how='inner')
    cj = cj.to_frame().join(cjbootstrap, how='inner')
    return cj, results