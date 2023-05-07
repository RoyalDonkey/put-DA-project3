#!/usr/bin/env python3

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from typing import Tuple
import helper_layers
import helper_training
import helper_explainability


CSV = "../data/car-evaluation.csv"
CSV_COLNAMES = ("buying", "maint", "doors", "persons", "lug_boot", "safety", "class")
CRITERIA_NAMES = CSV_COLNAMES[:-1]
NO_CRITERIA = len(CSV_COLNAMES) - 1
IDEAL_ALTERNATIVE     = np.array([1., 1., 1., 1., 1., 1., 4.])
ANTIIDEAL_ALTERNATIVE = np.array([0., 0., 0., 0., 0., 0., 1.])
# drop "label" - included in definition for posterity
IDEAL_ALTERNATIVE     = IDEAL_ALTERNATIVE[:-1]
ANTIIDEAL_ALTERNATIVE = ANTIIDEAL_ALTERNATIVE[:-1]
MODEL_PATH = "UTA_model.pt2"
EVAL_RATIO = 0.1      # The percentage of all data that will be used for evaluation


class MonotonicDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.Tensor(features)
        self.labels = torch.Tensor(labels)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.features)


class UTA(nn.Sequential):
    def __init__(self, criteria_nr: int, hidden_nr: int, **kwargs):
        super().__init__()
        self.criterionLayerSpread = helper_layers.CriterionLayerSpread(
            criteria_nr, hidden_nr, **kwargs
        )
        self.activate_function = helper_layers.LeakyHardSigmoid(**kwargs)
        self.criterionLayerCombine = helper_layers.CriterionLayerCombine(
            criteria_nr, hidden_nr, **kwargs
        )
        self.sum_layer = helper_layers.SumLayer(criteria_nr, **kwargs)

    def forward(self, x, *args, **kwargs):
        for module in self._modules.values():
            x = module(x)
        return x


def load_data(fpath: str, colnames: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(fpath, header=None, names=colnames)
    print(df.head())
    data = df.to_numpy(dtype=np.float32)
    features = data[:, :-1]
    # Transforming the two cost-type criteria
    # Their values will now be 1 for the best performance, and 0 for the worst
    # This is done as a workaround!
    # Now nondecreasing marginal value functions can be used as intended
    features[:, 0] = 1 - features[:, 0]
    features[:, 1] = 1 - features[:, 1]

    labels = data[:, -1].astype(int)
    # Binarising the labels
    # Adjust when changing data to ensure the binarisation makes sense!
    labels = np.where(labels > 1, 1, 0)
    features = features.reshape(-1, 1, NO_CRITERIA)
    return features, labels


def run() -> None:
    print()
    print("Task 2: ANN-UTADIS")
    features, labels = load_data(CSV, CSV_COLNAMES)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=EVAL_RATIO,
        random_state=1234,  # arbitrary seed for reproducibility
        stratify=labels
    )
    train_loader = DataLoader(MonotonicDataset(X_train, y_train),
                              batch_size=len(X_train))
    test_loader  = DataLoader(MonotonicDataset(X_test, y_test),
                              batch_size=len(X_test))
    uta = UTA(NO_CRITERIA, 12)
    model = helper_layers.NormLayer(uta, NO_CRITERIA,
                                    IDEAL_ALTERNATIVE, ANTIIDEAL_ALTERNATIVE)
    best_acc, test_acc, best_auc, test_auc, best_f1, test_f1 = \
        helper_training.Train(model, train_loader, test_loader, MODEL_PATH)
    torch.save(model, "newer_ENTIRE_UTA.pt2")
    print("BEST ACC:", round(best_acc, 4))
    print("TEST ACC:", round(test_acc, 4))
    print("BEST AUC:", round(best_auc, 4))
    print("TEST AUC:", round(test_auc, 4))
    print("BEST F1:", round(best_f1, 4))
    print("TEST F1:", round(test_f1, 4))
    print("Showing plots of network's marginal value functions...")
    helper_explainability.plot_marginal_functions(model, CRITERIA_NAMES, NO_CRITERIA)


if __name__ == "__main__":
    run()
