"""Module with utility functions used in streamlit-rdatasets

Functions:
- load_data(...) -- Loads data located at a url string and returns data
as a Pandas DataFrame.
- cosine_similarity(...) -- Computes cosine similarity between two vectors
- intersect_dicts(...) -- Returns intersection of two dictionaries.
"""

import os
import pandas as pd
import numpy as np
import streamlit as st

from numba import jit
from typing import List
from typing import Union


@st.cache(allow_output_mutation=True, show_spinner=False)
def load_data(url: str,
              cols: List[str] = None,
              index_col: Union[int, str] = None) -> pd.DataFrame:
    """Loads and returns data located at url as a Pandas DataFrame.
    Data must be a .csv, .tsv, or .dta file.

    Args:
        url (str): 
            url string to data.

        cols (list of str): 
            List of selected column names; defaults to None.
            If None, then all columns are selected.

        index_col (int or str): 
            Column to set as index.

    Returns:
        DataFrame of selected columns.
    """
    ext = os.path.splitext(url)[1]
    if ext == '.csv':
        data = pd.read_csv(url, index_col=index_col)
    elif ext == '.tsv':
        data = pd.read_csv(url,
                           index_col=index_col,
                           sep='/t')
    elif ext == '.dta':
        data = pd.read_stata(url, index_col=index_col)
    if cols:
        data = data[cols]
    return data


@jit(nopython=True)
def normalise_vector(u: np.ndarray, order: int) -> np.ndarray:
    """Scales vector using l1 or l2 norms.
    """
    norm_u = u / np.linalg.norm(u, order)
    return norm_u


@jit(nopython=True)
def cosine_similarity(u: np.ndarray,
                      v: np.ndarray) -> np.ndarray:
    """Computes cosine similarity between two vectors.
    """
    uv = 0
    uu = 0
    vv = 0
    for i in range(u.shape[0]):
        uv += u[i]*v[i]
        uu += u[i]*u[i]
        vv += v[i]*v[i]
    cos_theta = 0
    if uu != 0 and vv != 0:
        cos_theta = uv/np.sqrt(uu*vv)
    return cos_theta


def intersect_dicts(a, b):
    return {k: a[k] for k in a.keys() & b.keys()}


if __name__ == "__main__":
    pass
