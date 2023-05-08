#!/usr/bin/env python3

import dalex
import torch
import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Callable, Optional
from task2 import UTA          # noqa: F401
from task3 import nonlinearNN  # noqa: F401

CSV = "../data/car-evaluation.csv"
CSV_COLNAMES = ("buying", "maint", "doors", "persons", "lug_boot", "safety", "class")
CRITERIA_NAMES = CSV_COLNAMES[:-1]
NO_CRITERIA = len(CSV_COLNAMES) - 1


def load_data(fpath: str, colnames: list[str], transform_cost: bool = False) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(fpath, header=None, names=colnames)
    # Binarize classes
    df[CSV_COLNAMES[-1]] = df[CSV_COLNAMES[-1]].map(lambda x: 1 if x >= 2. else 0)
    features = df[[*CSV_COLNAMES[:-1]]]
    labels = df[[CSV_COLNAMES[-1]]]

    if transform_cost:
        # Transforming the two cost-type criteria
        # Their values will now be 1 for the best performance, and 0 for the worst
        # This is done as a workaround!
        # Now nondecreasing marginal value functions can be used as intended
        features["buying"] = 1 - features["buying"]
        features["maint"] = 1 - features["maint"]
    return features, labels


def pytorch_pred(model, data: pd.DataFrame, model_forward: Optional[Callable] = None) -> np.ndarray:
    """DALEX does not support PyTorch natively and needs a custom predictor function that maps a (model, dataset) pair to an array of corresponding labels."""
    answers = []
    if model_forward is None:
        model_forward = model.forward
    for _, features in data.iterrows():
        x = torch.Tensor(np.array(features)).view(-1, 1, NO_CRITERIA)
        out = model_forward(x)
        answers.append(out.detach().numpy().flatten()[0])
    return np.array(answers)


def dalex_cateris_paribus(model: torch.nn.Module | xgb.Booster, features: pd.DataFrame, labels: pd.DataFrame, explain_indices: list[int], **kwargs) -> None:
    """Generates Cateris Paribus profiles for each queried alternative."""
    explain_alternatives = features.iloc[explain_indices]
    explainer = dalex.Explainer(model, features, labels, **kwargs)
    for _, alternative in explain_alternatives.iterrows():
        rf_profile = explainer.predict_profile(new_observation=alternative)
        rf_profile.plot()

def dalex_variable_attributions(model: torch.nn.Module | xgb.Booster, features: pd.DataFrame, labels: pd.DataFrame, explain_indices: list[int], **kwargs) -> None:
    """Breaks down variable attributions for each queried alternative."""
    explain_alternatives = features.iloc[explain_indices]
    explainer = dalex.Explainer(model, features, labels, **kwargs)
    for i in range(len(explain_indices)):
        alternative = explain_alternatives.iloc[i]
        rf_profile = explainer.predict_parts(alternative, type='break_down_interactions', label=('worst', 'medium', 'best')[i], random_state=1234)
        rf_profile.plot()

def dalex_variable_importances(model: torch.nn.Module | xgb.Booster, features: pd.DataFrame, labels: pd.DataFrame, **kwargs) -> None:
    """Computes the importance of each variable for the entire model."""
    explainer = dalex.Explainer(model, features, labels, **kwargs)
    rf_profile = explainer.model_parts(loss_function='auc', type='variable_importance', N=None, random_state=1234)
    rf_profile.plot()


if __name__ == "__main__":
    # MODEL_TYPE
    #   1:  XGBoost
    #   2:  UTA-ANN
    #   3:  nonlinear-ANN
    # TASK_TYPE
    #   1:  cateris paribus
    #   2:  variable attributions
    #   3:  variable importances (ignores EXPLAIN_INDICES)
    MODEL_TYPE = 0
    TASK_TYPE = 0
    EXPLAIN_INDICES = [0, 1635, 1727]

    #############  ! CONFIGURATION ABOVE !  #############

    model: torch.nn.Module | xgb.Booster
    if MODEL_TYPE == 1:
        model = xgb.Booster()
        model.load_model("xgboost.model")
    elif MODEL_TYPE == 2:
        model = torch.load("IMPORTANTEST_ENTIRE_UTA.pt2")
    elif MODEL_TYPE == 3:
        model = torch.load("ANN_model.pt2")
    else:
        assert False

    features, labels = load_data(CSV, list(CSV_COLNAMES), MODEL_TYPE != 1)
    kwargs = {'predict_function': pytorch_pred} if MODEL_TYPE != 1 else {}

    if TASK_TYPE == 1:
        dalex_cateris_paribus(model, features, labels, EXPLAIN_INDICES, **kwargs)
    elif TASK_TYPE == 2:
        dalex_variable_attributions(model, features, labels, EXPLAIN_INDICES, **kwargs)
    elif TASK_TYPE == 3:
        dalex_variable_importances(model, features, labels, **kwargs)
    else:
        assert False
