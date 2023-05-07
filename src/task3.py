import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import pandas as pd
from typing import Tuple
import matplotlib.pyplot as plt


CSV = "../data/car-evaluation.csv"
CSV_COLNAMES = ("buying", "maint", "doors", "persons", "lug_boot", "safety", "class")
CRITERIA_NAMES = CSV_COLNAMES[:-1]
NO_CRITERIA = len(CSV_COLNAMES) - 1
IDEAL_ALTERNATIVE     = np.array([1., 1., 1., 1., 1., 1., 4.])
ANTIIDEAL_ALTERNATIVE = np.array([0., 0., 0., 0., 0., 0., 1.])
# drop "label" - included in definition for posterity
IDEAL_ALTERNATIVE     = IDEAL_ALTERNATIVE[:-1]
ANTIIDEAL_ALTERNATIVE = ANTIIDEAL_ALTERNATIVE[:-1]
MODEL_PATH = "NN_model.pt2"


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


def load_data(fpath: str, colnames: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(fpath, header=None, names=colnames)
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


class nonlinearNN(nn.Module):
    def __init__(self, hidden_layers: int, no_criteria: int):
        super().__init__()
        self.out_activation = nn.Sigmoid()
        self.layers = nn.Sequential()
        # process individual criteria
        for _ in range(hidden_layers):
            self.layers.append(nn.Linear(no_criteria, no_criteria))
            self.layers.append(nn.LeakyReLU())
        self.layers.append(nn.Linear(no_criteria, 1))

    def forward(self, x: torch.Tensor):
        for module in self._modules.values():
            x = module(x)
        x = self.out_activation(x)
        x = x.view(-1)
        return x


def train(model: nonlinearNN, train_dataloader: DataLoader, test_dataloader: DataLoader,
          save_path: str, lr: float = 0.01, epoch_nr: int = 200):
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99))
    best_acc = 0.0
    best_auc = 0.0
    best_f1 = 0.0
    train_loss = []
    for epoch in tqdm(range(epoch_nr)):
        for data in train_dataloader:
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            train_loss.append(loss.detach().mean())
            loss.backward()
            optimizer.step()
            binary_outputs = np.where(outputs.detach().numpy() > 0.5, 1, 0)
            binary_labels = labels.detach().numpy().astype(int)
            acc = accuracy_score(binary_labels, binary_outputs)
            auc = roc_auc_score(binary_labels, binary_outputs)
            f1 = f1_score(binary_labels, binary_outputs)
        if acc > best_acc:
            best_acc = acc
            best_auc = auc
            best_f1 = f1
            with torch.no_grad():
                for data in test_dataloader:
                    inputs, labels = data
                    outputs = model(inputs)
                    loss_test = loss_fn(outputs, labels)
                    binary_outputs = np.where(outputs.detach().numpy() > 0.5, 1, 0)
                    binary_labels = labels.detach().numpy().astype(int)
                    acc_test = accuracy_score(binary_labels, binary_outputs)
                    auc_test = roc_auc_score(binary_labels, binary_outputs)
                    f1_test = f1_score(binary_labels, binary_outputs)
        torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_train": loss,
                    "loss_test": loss_test,
                    "accuracy_train": acc,
                    "accuracy_test": acc_test,
                    "auc_train": auc,
                    "auc_test": auc_test,
                    "f1_train": f1,
                    "f1_test": f1_test,
                },
                save_path,
            )
    plt.plot(train_loss)
    plt.show()
    return best_acc, acc_test, best_auc, auc_test, best_f1, f1_test


def run() -> None:
    print()
    print('Task 3: Neural network w/ nonlinear activation functions')
    features, labels = load_data(CSV, CSV_COLNAMES)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2)
    train_loader = DataLoader(MonotonicDataset(X_train, y_train),
                              batch_size=len(X_train))
    test_loader  = DataLoader(MonotonicDataset(X_test, y_test),
                              batch_size=len(X_test))
    model = nonlinearNN(12, NO_CRITERIA)
    best_acc, test_acc, best_auc, test_auc, best_f1, test_f1 = \
        train(model, train_loader, test_loader, "ANN_model.pt2")
    print("BEST ACC:", round(best_acc, 4))
    print("TEST ACC:", round(test_acc, 4))
    print("BEST AUC:", round(best_auc, 4))
    print("TEST AUC:", round(test_auc, 4))
    print("BEST F1:", round(best_f1, 4))
    print("TEST F1:", round(test_f1, 4))


if __name__ == "__main__":
    run()
