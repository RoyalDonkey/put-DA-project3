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




if __name__ == "__main__":
    # MODEL_TYPE
    #   1:  XGBoost
    #   2:  UTA-ANN
    #   3:  nonlinear-ANN
    #   Task: evaluate alternatives from file, loaded based on their index
    MODEL_TYPE = 3
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

    if MODEL_TYPE == 1:
        for idx in EXPLAIN_INDICES:
            np_row = np.array(features.iloc[idx])
            print(f"xgboost output for {np_row}, {labels.iloc[idx]['class']}: \
                   {model.predict(features[idx])}")
    elif MODEL_TYPE == 2:
        for idx in EXPLAIN_INDICES:
            np_row = np.array(features.iloc[idx])
            x = torch.Tensor(np_row).view(-1, 1, NO_CRITERIA)
            print(f"ANN-UTADIS output for {np_row}, "
                  + f"{labels.iloc[idx]['class']}: {model.forward(x)[0][0]}")
    elif MODEL_TYPE == 3:
        for idx in EXPLAIN_INDICES:
            np_row = np.array(features.iloc[idx])
            x = torch.Tensor(np_row).view(-1, 1, NO_CRITERIA)
            print(f"Nonlinear ANN output for {np_row}, "
                  + f"{labels.iloc[idx]['class']}: {model.forward(x)[0]}")
    else:
        assert False
