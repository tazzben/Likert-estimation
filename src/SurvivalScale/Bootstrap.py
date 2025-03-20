from multiprocessing import Pool
import numpy as np
import pandas as pd
from tqdm import tqdm
from .Solver import solver

def bootstrap(pdData, columns=None, n_bootstraps=1000, alpha=0.05, block_id=None):
    if not columns:
        columns = pdData.columns[3:]

    def bootstrap_iteration(_):
        if block_id is None:
            sample_data = pdData.sample(n=len(pdData), replace=True)
        else:
            unique_blocks = pdData[block_id].unique()
            sampled_blocks = np.random.choice(unique_blocks, size=len(unique_blocks), replace=True)
            sample_data = pd.concat([pdData[pdData[block_id] == block] for block in sampled_blocks])
        cjResults, solvedParams = solver(sample_data, columns)
        return cjResults, solvedParams

    return final_results