"""Functions for preprocessing data and performing AutoML.
"""

import pandas as pd

from typing import List
from typing import Union
from typing import Mapping


def _factor_wrangler(
    data: pd.DataFrame,
    is_factor: Union[None, List[str]],
    categories: Union[None, Mapping[str, List[Union[str, int, float]]]] = None,
    str_to_cat: bool = True,
) -> pd.DataFrame:
    """Converts columns in `is_factor` to `CategoricalDtype`.
    If `str_to_cat` is set to True, converts all `StringDtype` columns
    to `CategoricalDtype`.
    """
    cat_cols = []
    if str_to_cat:
        str_cols = (data.select_dtypes(include=['string'])
                        .columns
                        .tolist())
        cat_cols += str_cols
    if is_factor:
        cat_cols += is_factor
    if cat_cols:
        for col in cat_cols:
            data.loc[:, col] = (data.loc[:, col]
                                    .astype('category'))
    return data


def _na_wrangler(data: pd.DataFrame):
    """Runs MICE algorithm on data to fill in missing values.
    """
    return data


def clean_data(
    data: pd.DataFrame,
    is_factor: Union[None, List[str]] = None,
    categories: Union[None, Mapping[str, List[Union[str, int, float]]]] = None,
    str_to_cat: bool = True,
) -> pd.DataFrame:
    """Data preprocessing pipeline.
    Runs the following data wranglers on `data`:
    1. _factor_wrangler
    2. _na_wrangler

    Note: any missing values that remain after `clean_data` are
    
    """
    data = (data.pipe(_factor_wrangler, is_factor, categories, str_to_cat)
                .pipe(_na_wrangler))
    return data


if __name__ == "__main__":
    pass
