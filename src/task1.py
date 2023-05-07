#!/usr/bin/env python3
"""
Task 1: XDGBoost
https://xgboost.readthedocs.io/en/latest/python/python_intro.html
"""
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from numpy import round
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from typing import Any

CSV = '../data/car-evaluation.csv'
CSV_COLNAMES = ('buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class')
NUM_BOOST_ROUND = 10  # Number of tree boosting rounds
EVAL_RATIO = 0.1      # The percentage of all data that will be used for evaluation


def load_data(fpath: str) -> tuple[xgb.DMatrix, xgb.DMatrix]:
    print(f'-> Loading data from {fpath}')
    df = pd.read_csv(CSV, header=None, names=CSV_COLNAMES)
    # Binarize classes
    df[CSV_COLNAMES[-1]] = df[CSV_COLNAMES[-1]].map(lambda x: 1 if x >= 2. else 0)
    print(df.head())
    features = df[[*CSV_COLNAMES[:-1]]]
    labels = df[[CSV_COLNAMES[-1]]]

    print('-> Splitting data into training and evaluation sets')
    train_data, eval_data, train_labels, eval_labels = train_test_split(
        features, labels,
        test_size=EVAL_RATIO,
        random_state=1234,  # arbitrary seed for reproducibility
        stratify=labels
    )

    # Print some statistics for sanity
    train_0s = train_labels[train_labels[CSV_COLNAMES[-1]] == 0]
    train_1s = train_labels[train_labels[CSV_COLNAMES[-1]] == 1]
    print(f'training set:    {len(train_data)} total, {len(train_1s)} 1\'s, {len(train_0s)} 0\'s')
    eval_0s = eval_labels[eval_labels[CSV_COLNAMES[-1]] == 0]
    eval_1s = eval_labels[eval_labels[CSV_COLNAMES[-1]] == 1]
    print(f'evaluation set:  {len(eval_data)} total, {len(eval_1s)} 1\'s, {len(eval_0s)} 0\'s')
    assert len(train_1s) != 0
    assert len(train_0s) != 0
    assert len(eval_1s) != 0
    assert len(eval_0s) != 0

    dtrain = xgb.DMatrix(train_data, label=train_labels)
    deval  = xgb.DMatrix(eval_data, label=eval_labels)
    return dtrain, deval

def configure() -> dict[str, Any]:
    # https://xgboost.readthedocs.io/en/latest/parameter.html

    # Global configuration
    print('-> Configuring XGBoost global parameters')
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
        'eval_metric'          : ['auc', 'error'],
        'seed'                 : 0,
    }

def evaluate(bst: xgb.Booster, evals_result: dict[Any, Any], deval: xgb.DMatrix) -> None:
    print('-> Evaluating the trained model')
    print('eval set Accuracy:', 1. - evals_result['deval']['error'][-1])
    print('eval set AUC:     ', evals_result['deval']['auc'][-1])

    # Unfortunately, xgb doesn't seem to provide a built-in way to calculate
    # the F1 score, so we need to run the model once more to do it manually:
    pred = bst.predict(deval)
    print('eval set F1:      ', f1_score(deval.get_label(), round(pred)))

def plot(bst: xgb.Booster) -> None:
    print('-> Generating plots')
    fig, ax = plt.subplots(figsize=(15, 15))
    xgb.plot_tree(bst, ax=ax, num_trees=0)
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 15))
    xgb.plot_importance(bst, ax=ax)
    plt.show()

def run() -> None:
    print()
    print('Task 1: XDGBoost')

    dtrain, deval = load_data(CSV)
    params = configure()

    evallist = [(deval, 'deval')]
    evals_result: dict[Any, Any] = {}
    bst = xgb.train(params, dtrain, NUM_BOOST_ROUND, evals=evallist, evals_result=evals_result)

    evaluate(bst, evals_result, deval)
    plot(bst)

if __name__ == '__main__':
    run()
