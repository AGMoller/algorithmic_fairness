import numpy as np
import pandas as pd
from typing import List
import scipy
from tqdm import tqdm


def reproject_features(
        data: pd.DataFrame, protected_cols: List[str], nonprotected_cols: List[str]) -> np.ndarray:
    """
    generate a fair representation of nonprotected columns which are independent from any columns in protected_cols

    data : pd.DataFrame
        dataframe with columns to be projected
    protected_cols : List[str]
        list of protected columns
    nonprotected_cols : List[str]
        list of non-protected columns
    """
    # make a copy of data
    df: pd.DataFrame = data.copy()
    # df is our data
    # extract data about protected columns
    nonprotect: np.ndarray = df[nonprotected_cols].values
    protect: np.ndarray = df[protected_cols].values
    # extract data about nonprotected columns
    debiased_nonprotect: np.ndarray = df[nonprotected_cols].values
    # crease an orthonormal basis
    base_protect: np.ndarray = scipy.linalg.orth(protect)

    batch_size = 1000

    # go through all protected attributes and calculate their contribution to
    # the reprojection to the hyperplane
    debiased_data = list()
    for j in range(debiased_nonprotect.shape[1]):
        for index in range(0, data.shape[0], batch_size):
            start, stop = index, min(index + batch_size, data.shape[0])
            debiased_nonprotect[start:stop,
                                j] -= base_protect[start:stop] @ base_protect[start:stop].T @ nonprotect[start:stop, j]
    return debiased_nonprotect


def reproject_features_w_regul(
        data: pd.DataFrame, protected_cols: List[str], nonprotected_cols: List[str], lambda_: float) -> np.ndarray:
    """
    generate a fair representation of nonprotected columns which are independent from any columns in protected_cols
    data: a data frame
    protected_cols: list of strings, the protected columns
    nonprotected_col: string, all other data columns
    lambda_: float number between 0 and 1, 0 means totally fair; 1 means same as raw data
    """

    # run the normal reproject_features function
    r: np.ndarray = reproject_features(data, protected_cols, nonprotected_cols)

    # extract data about nonprotected variables
    nonprotect: np.ndarray = data[nonprotected_cols].values
    # standardize columns

    return r + lambda_ * (nonprotect - r)
