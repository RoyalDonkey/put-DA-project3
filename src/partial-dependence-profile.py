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


def pred(model, data: pd.DataFrame, model_forward: Optional[Callable] = None) -> np.ndarray:
    answers = []
    if model_forward is None:
        model_forward = model.forward
    for _, features in data.iterrows():
        x = torch.Tensor(np.array(features)).view(-1, 1, NO_CRITERIA)
        out = model_forward(x)
        answers.append(out.detach().numpy().flatten()[0])
    return np.array(answers)

def dalex_profile(model: torch.nn.Module | xgb.Booster, features: pd.DataFrame, labels: pd.DataFrame, explain_indices: list[int], **kwargs) -> None:
    explain_alternatives = features.iloc[explain_indices]
    explainer = dalex.Explainer(model, features, labels, **kwargs)
    for i in range(len(explain_indices)):
        alternative = explain_alternatives.iloc[i]
        rf_profile = explainer.predict_parts(alternative, type='break_down_interactions', label=('worst', 'medium', 'best')[i])
        rf_profile.plot()

if __name__ == "__main__":
    features, labels = load_data(CSV, list(CSV_COLNAMES))

    model: torch.nn.Module | xgb.Booster
    explain_indices = [0, 1635, 1727]

    # XGBoost
    model = xgb.Booster()
    model.load_model("xgboost.model")
    dalex_profile(model, features, labels, explain_indices)

    features, labels = load_data(CSV, list(CSV_COLNAMES), True)

    # UTA-ANN
    model = torch.load("IMPORTANTEST_ENTIRE_UTA.pt2")
    dalex_profile(model, features, labels, explain_indices, predict_function=pred)

    # nonlinear-ANN
    model = torch.load("ANN_model.pt2")
    dalex_profile(model, features, labels, explain_indices, predict_function=pred)
