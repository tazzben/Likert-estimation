import numpy as np
import pandas as pd
from scipy.stats.distributions import chi2
from .Solver import solver, restricted_solver
from .Marginal import marginalize_df, marginalize_cjdf
from .Bootstrap import bootstrap

def compareKBound(x):
    return pd.to_numeric(x, downcast='integer')

def get_results(data, columns=None, bootstrap_iterations=1000, alpha=0.05, block_id=None):
    """
    Get the results of the SurvivalScale analysis.

    Parameters:
    - data (pd.DataFrame): The input data containing the features and other necessary columns.
    - columns (list, optional): The list of feature columns to be used in the analysis. 
                                 If None, all columns except 'k', 'question', 'bound' will be used.
    - bootstrap_iterations (int, optional): The number of bootstrap iterations to perform. Default is 1000.
    - alpha (float, optional): The significance level for the bootstrap confidence intervals. Default is 0.05.
    - block_id (str, optional): The column name to use for blocking in the bootstrap analysis. Default is None.

    Returns:
    - pd.DataFrame: question-level results (Cj values).
    - pd.DataFrame: variable-level results (beta values and their marginals).
    - pd.DataFrame: metrics including McFadden pseudo R-squared, log-likelihood, AIC, BIC, etc. as keys in a DataFrame.
    """
    if columns is None:
        columns = [col for col in data.columns if col not in ['k', 'question', 'bound']]
    # Ensure data contains the necessary columns
    required_columns = ['k', 'question', 'bound']
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Data must contain the column '{col}'")
    
    # Ensure 'k', 'question', and 'bound' are the first three columns
    data = data[['k', 'question', 'bound'] + [col for col in data.columns if col not in ['k', 'question', 'bound']]]

    # Check if the specified columns are in the data
    for col in columns:
        if col not in data.columns:
            raise ValueError(f"Specified column '{col}' is not in the data")
    # Check if block_id is in the data if provided
    if block_id is not None and block_id not in data.columns:
        raise ValueError(f"Specified block_id '{block_id}' is not in the data")
    # Preprocess the data
    # Drop rows with NaN values
    # Convert 'k' and 'bound' to numeric, ensuring they are valid integers
    data.dropna(inplace=True)
    data = data[pd.to_numeric(data['k'], errors='coerce').notnull()]
    data = data[pd.to_numeric(data['bound'], errors='coerce').notnull()]
    data[["k", "bound"]] = data[["k", "bound"]].apply(compareKBound)
    data = data[data['k'] <= data['bound']]

    # Ensure columns are numeric
    data[columns] = data[columns].apply(pd.to_numeric, errors='coerce')
    data.dropna(inplace=True)
    # Check if there are any rows left after preprocessing
    if data.empty:
        raise ValueError("No valid data left after preprocessing. Please check your input data.")
    # Solve the model
    cj, beta, fun, parlen = solver(data, columns=columns)
    cj.index.name = None
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
    # Get cj marginals
    cjmarginals = marginalize_cjdf(data, beta, cj, columns=columns, discrete=False)
    cjmarginalsD = marginalize_cjdf(data, beta, cj, columns=columns, discrete=True)
    cjmarginalsD = cjmarginalsD.rename(columns={'marginal': 'marginal_discrete'})
    cj = cj.to_frame().join(cjmarginals, how='inner')
    cj = cj.join(cjmarginalsD, how='inner')
    # Join bootstrap results with cj
    cj = cj.join(cjbootstrap, how='inner')
    # Calculate McFadden pseudo R-squared, log-likelihood, AIC, BIC, etc.
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

    # Create a clean pandas DataFrame from metrics
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    metrics_df.set_index('Metric', inplace=True, drop=True)
    metrics_df.index.name = None
    return cj, results, metrics_df
