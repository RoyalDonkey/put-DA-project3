import torch
from sklearn.metrics import roc_auc_score, f1_score
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Tuple
import numpy as np


def Regret(x: torch.Tensor, target: torch.Tensor) -> float:
    return torch.mean(
        torch.relu(-(target >= 1).float() * x) + torch.relu((target < 1).float() * x)
    )


def Accuracy(x: torch.Tensor, target: torch.Tensor) -> float:
    return (target == (x[:, 0] > 0) * 1).detach().numpy().mean()


def AUC(x: torch.Tensor, target: torch.Tensor) -> float:
    return roc_auc_score(target.detach().numpy(), x.detach().numpy()[:, 0])


def F1(x: torch.Tensor, target: torch.Tensor) -> float:
    # Without binarisation, the function doesn't pass
    # I'm not sure if that's the wat to do this though
    binary_x = np.where(x.detach().numpy()[:, 0] > 0, 1, 0)
    return f1_score(target.detach().numpy(), binary_x)


def Train(model, train_dataloader: DataLoader, test_dataloader: DataLoader,
          save_path: str, lr: float = 0.01, epoch_nr: int = 200
          ) -> Tuple[float, float, float, float, float, float]:
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99))
    best_acc = 0.0
    best_auc = 0.0
    best_f1 = 0.0
    for epoch in tqdm(range(epoch_nr)):
        for data in train_dataloader:
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = Regret(outputs, labels)
            loss.backward()
            optimizer.step()
            acc = Accuracy(outputs, labels)
            auc = AUC(outputs, labels)
            f1 = F1(outputs, labels)
        if acc > best_acc:
            best_acc = acc
            best_auc = auc
            best_f1 = f1
            with torch.no_grad():
                for data in test_dataloader:
                    inputs, labels = data
                    outputs = model(inputs)
                    loss_test = Regret(outputs, labels)
                    acc_test = Accuracy(outputs, labels)
                    auc_test = AUC(outputs, labels)
                    f1_test = F1(outputs, labels)

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

    return best_acc, acc_test, best_auc, auc_test, best_f1, f1_test
