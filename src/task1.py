"""
Task 1: XDGBoost
https://xgboost.readthedocs.io/en/latest/python/python_intro.html
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Any

CSV = '../data/breast-cancer.csv'
CSV_COLNAMES = ('F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'class')


def load_data(fpath: str) -> xgb.DMatrix:
    print(f'-> Loading data from {fpath}')
    df = pd.read_csv(CSV, header=None, names=CSV_COLNAMES)
    print(df.head())
    data = df.to_numpy(dtype=np.float32)
    features = data[:, :-1]
    labels = data[:, -1]
    return xgb.DMatrix(features, label=labels)

def configure() -> dict[str, Any]:
    # https://xgboost.readthedocs.io/en/latest/parameter.html

    # Global configuration
    xgb.set_config(verbosity=2, use_rmm=True)

    # https://xgboost.readthedocs.io/en/latest/tutorials/monotonic.html
    # The monotone values below are arbitrary guesses
    monotone_constraints = {
        'F1': 1,
        'F2': -1,
        'F3': 1,
        'F4': -1,
        'F5': 1,
        'F6': -1,
        'F7': 1,
    }

    return {
        'booster'              : 'gbtree',
        'learning_rate'        : 0.3,
        'min_split_loss'       : 0.0,
        'max_depth'            : 6,
        'min_child_weight'     : 1,
        'max_delta_step'       : 0.0,
        'subsample'            : 1.0,
        'sampling_method'      : 'uniform',
        'reg_lambda'           : 1.0,
        'reg_alpha'            : 0.0,
        'tree_method'          : 'auto',
        'process_type'         : 'default',
        'grow_policy'          : 'depthwise',
        'max_leaves'           : 0,
        'predictor'            : 'auto',
        'num_parallel_tree'    : 1,
        'monotone_constraints' : monotone_constraints,
    }

def run() -> None:
    print()
    print('Task 1: XDGBoost')

    dm = load_data(CSV)

    params = configure()
