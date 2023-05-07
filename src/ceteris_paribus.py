import dalex
import torch
import pandas as pd
import numpy as np
from typing import Tuple, Callable
from task2 import UTA

CSV = "../data/car-evaluation.csv"
CSV_COLNAMES = ("buying", "maint", "doors", "persons", "lug_boot", "safety", "class")
CRITERIA_NAMES = CSV_COLNAMES[:-1]
NO_CRITERIA = len(CSV_COLNAMES) - 1
IDEAL_ALTERNATIVE     = np.array([1., 1., 1., 1., 1., 1., 4.])
ANTIIDEAL_ALTERNATIVE = np.array([0., 0., 0., 0., 0., 0., 1.])
# drop "label" - included in definition for posterity
IDEAL_ALTERNATIVE     = IDEAL_ALTERNATIVE[:-1]
ANTIIDEAL_ALTERNATIVE = ANTIIDEAL_ALTERNATIVE[:-1]


def load_data(fpath: str, colnames: str, transform_cost: bool = False) -> Tuple[np.ndarray, np.ndarray]:
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


def pred(model, data: pd.DataFrame, model_forward: Callable = None) -> np.ndarray:
    answers = []
    if model_forward is None:
        model_forward = model.forward
    for _, features in data.iterrows():
        x = torch.Tensor(np.array(features)).view(-1, 1, NO_CRITERIA)
        out = model_forward(x)
        answers.append(out.detach().numpy().flatten()[0])
    return np.array(answers)


if __name__ == "__main__":
    features, labels = load_data(CSV, CSV_COLNAMES, True)
    print(features)
    explain_indices = [0, 1635, 1727]
    explain_alternatives = features.iloc[explain_indices]
    model = torch.load("ENTIRE_UTA.pt2")
    explainer = dalex.Explainer(model, features, labels, predict_function=pred)
    for _, alternative in explain_alternatives.iterrows():
        rf_profile = explainer.predict_profile(new_observation = alternative)
        rf_profile.plot()
