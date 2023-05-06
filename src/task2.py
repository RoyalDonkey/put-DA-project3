import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from typing import Tuple
import helper_layers
import helper_training


CSV = "../data/car-evaluation.csv"
# CSV_COLNAMES = ('F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'class')
CSV_COLNAMES = ("buying", "maint", "doors", "persons", "lug_boot", "safety", "class")
NO_CRITERIA = len(CSV_COLNAMES) - 1
IDEAL_ALTERNATIVE     = np.array([0., 0., 1., 1., 1., 1., 4.])
ANTIIDEAL_ALTERNATIVE = np.array([1., 1., 0., 0., 0., 0., 1.])
# drop "label" - included in definition for posterity
IDEAL_ALTERNATIVE     = IDEAL_ALTERNATIVE[:-1]
ANTIIDEAL_ALTERNATIVE = ANTIIDEAL_ALTERNATIVE[:-1]
MODEL_PATH = "UTA_model.pt2"


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
    labels = data[:, -1].astype(int)
    labels = np.where(labels > 2, 1, 0)
    features = features.reshape(-1, 1, NO_CRITERIA)
    return features, labels


def run() -> None:
    print()
    print('Task 2: Ch-Constr/UTADIS')
    # TODO
    # figure out the hook thing
    # likely split this into separate model building and visualisation scripts

    features, labels = load_data(CSV, CSV_COLNAMES)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2)
    train_loader = DataLoader(MonotonicDataset(X_train, y_train),
                              batch_size=len(X_train))
    test_loader  = DataLoader(MonotonicDataset(X_test, y_test),
                              batch_size=len(X_test))
    uta = UTA(NO_CRITERIA, 12)
    model = helper_layers.NormLayer(uta, NO_CRITERIA)
    best_acc, test_acc, best_auc, test_auc = helper_training.Train(
        model, train_loader, test_loader, MODEL_PATH)

    print("BEST ACC:", best_acc)
    print("TEST ACC:", test_acc)
    print("BEST AUC:", best_auc)
    print("TEST AUC:", test_auc)


if __name__ == "__main__":
    run()
