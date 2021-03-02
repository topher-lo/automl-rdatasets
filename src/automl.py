"""Functions for preprocessing data and performing AutoML.
"""

import pandas as pd

from typing import List
from typing import Union
from typing import Mapping
from tpot import TPOTClassifier

from sklearn.model_selection import train_test_split


def _factor_wrangler(
    data: pd.DataFrame,
    is_factor: Union[None, List[str]],
    is_ordered: Union[None, List[str]],
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
    # Set categories
    for col, cats in categories.items():
        data.loc[:, col] = (data.loc[:, col]
                                .cat
                                .set_categories(cats))
    # Set is_ordered
    for cat in is_ordered:
        data.loc[:, col] = (data.loc[:, col]
                                .cat
                                .as_ordered())
    return data


def clean_data(
    data: pd.DataFrame,
    is_factor: Union[None, List[str]] = None,
    is_ordered: Union[None, List[str]] = None,
    categories: Union[None, Mapping[str, List[Union[str, int, float]]]] = None,
    str_to_cat: bool = True,
) -> pd.DataFrame:
    """Data preprocessing pipeline.
    Runs the following data wranglers on `data`:
    1. _factor_wrangler
    """
    data = (data.pipe(_factor_wrangler, is_factor, categories, str_to_cat))
    return data


def encode_data(data: pd.DataFrame) -> pd.DataFrame:
    """Transforms columns with unordered `category` dtype
    using `pd.get_dummies`. Transforms columns with ordered `category`
    dtype using `series.cat.codes`.
    """
    cat_cols = (data.select_dtypes(include=['category']))
    unordered = [col.name for col in cat_cols if not(col.cat.ordered)]
    ordered = [col.name for col in cat_cols if col not in unordered]
    if cat_cols.any():
        data = pd.get_dummies(data, columns=unordered, dummy_na=True)
        data.iloc[:, ordered] = data.iloc[:, ordered].cat.codes
    return data


def run_automl(data: pd.DataFrame,
               outcome_col: str,
               ml_task: str,
               train_size: float,
               test_size: float,
               *args, **kwargs) -> str:
    """Runs AutoML using TPOT. Returns best pipeline found by TPOT as a string
    of Python code.
    """
    # Features and outcome columns
    features = [col for col in data.columns if outcome_col not in col]
    outcome = [col for col in data.columns if outcome_col in col]
    if len(outcome) == 1:
        outcome = outcome[0]
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, features],
                                                        data.iloc[:, outcome],
                                                        test_size,
                                                        train_size)
    # Optimise pipeline
    if ml_task == 'Classification':
        tpot = TPOTClassifier(**kwargs)
        tpot.fit(X_train, y_train)
    else:
        tpot = TPOTRegressor(**kwargs)
        tpot.fit(X_train, y_train)
    # Export Python code for best pipeline found
    pipeline_code = tpot.export()
    return pipeline_code



if __name__ == "__main__":
    pass
