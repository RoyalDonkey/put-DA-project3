"""
Task 1: XDGBoost
https://xgboost.readthedocs.io/en/latest/python/python_intro.html
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Any

CSV = '../data/car-evaluation.csv'
CSV_COLNAMES = ('buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class')


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
        'buying': -1,
        'maint': -1,
        'doors': 1,
        'persons': 1,
        'lug_boot': 1,
        'safety': 1,
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
        # https://xgboost.readthedocs.io/en/latest/parameter.html#learning-task-parameters
        # I half-guessed the following 2 values, I suppose they should be compliant with
        # the error function described in lab7 pdf on page 8:
        'objective'            : 'binary:logistic',
        'eval_metric'          : ['auc'],
        'seed'                 : 0,
    }

def run() -> None:
    print()
    print('Task 1: XDGBoost')

    dm = load_data(CSV)

    params = configure()
