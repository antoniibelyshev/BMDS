import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

from pbmds import PBMDS, PBMDSTrainer, PBMDSDataset
from classifier import NNClassifier, NNClassifierTrainer
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


def train_pbmds(
    train_dmat_sqr: Tensor,
    dataset_name: str,
    *,
    lr: float = 4e-4,
    epochs: int = 100,
) -> PBMDS:
    torch.manual_seed(0)
    dataset = PBMDSDataset(train_dmat_sqr)
    dataloader = DataLoader(dataset, batch_size=train_dmat_sqr.shape[0], shuffle=True, drop_last=True)
    pbmds = PBMDS(train_dmat_sqr.shape[0], train_dmat_sqr.shape[0])
    optimizer = torch.optim.Adam(pbmds.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.97, step_size=1)
    scheduler = None
    trainer = PBMDSTrainer(
        pbmds,
        None,
        optimizer,
        scheduler,
    )
    trainer.train(dataloader, epochs=epochs, name=f"{dataset_name}_pbmds")
    return trainer.ema_model


def train_classifier(
    x_train: Tensor,
    x_eval: Tensor,
    y_train: Tensor,
    y_eval: Tensor,
    dataset_name: str,
) -> tuple[float, float]:
    # classifier = NNClassifier(train_embeddings.shape[1], 2)
    # optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.8, step_size=10)
    # trainer = NNClassifierTrainer(
    #     classifier,
    #     None,
    #     optimizer,
    #     scheduler,
    # )
    # dataloader = DataLoader(
    #     TensorDataset(train_embeddings, train_labels), batch_size=1, shuffle=True, drop_last=True,
    # )
    # eval_dataloader = DataLoader(
    #     TensorDataset(eval_embeddings, eval_labels), batch_size=100, shuffle=False
    # )
    # trainer.train(
    #     dataloader,
    #     epochs=100,
    #     name=f"{dataset_name}_classifier",
    #     eval_dataloader=eval_dataloader,
    # )
    clf = MLPClassifier(solver='sgd', hidden_layer_sizes=(100, 100), random_state=0, max_iter=1000, n_iter_no_change=1000, learning_rate_init=1e-3, batch_size=16)
    clf.fit(x_train, y_train)
    
    train_acc = accuracy_score(y_train, clf.predict(x_train))
    eval_acc = accuracy_score(y_eval, clf.predict(x_eval))
    return train_acc, eval_acc


def main(dataset_name: str) -> None:
    data_file = np.load(f"data/{dataset_name}.npz")
    dmat_sqr = torch.tensor(data_file["pw_dmat"]).float().square()
    labels = torch.tensor(data_file["labels"])

    kfold_validation = KFold(n_splits=10, shuffle=True, random_state=0)
    train_accs = []
    eval_accs = []
    for i, (train_idx, eval_idx) in enumerate(kfold_validation.split(np.arange(len(dmat_sqr))), 1): # type: ignore
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_dmat_sqr = dmat_sqr[train_idx][:, train_idx].to(device)
        eval_dmat_sqr = dmat_sqr[eval_idx][:, train_idx].to(device)

        pbmds = train_pbmds(train_dmat_sqr, dataset_name)
        x_train = pbmds.embedding(train_dmat_sqr)
        x_eval = pbmds.embedding(eval_dmat_sqr)

        y_train = labels[train_idx]
        y_eval = labels[eval_idx]

        train_acc, eval_acc = train_classifier(
            x_train, x_eval, y_train, y_eval, dataset_name
        )
        print(f"Fold {i}: Train acc: {train_acc}, Eval acc: {eval_acc}")
        train_accs.append(train_acc)
        eval_accs.append(eval_acc)

        np.savez(
            f"tmp/fold_{i}.npz",
            train_dmat_sqr = train_dmat_sqr.cpu(),
            eval_dmat_sqr = eval_dmat_sqr.cpu(),
            x_train = x_train,
            x_eval = x_eval,
            y_train = y_train,
            y_eval = y_eval,
        )

    print("Train accs:")
    print(train_accs)
    print("Eval accs")
    print(eval_accs)
    print(f"Average train acc: {np.mean(train_accs)}, std: {np.std(train_accs)}")
    print(f"Average eval acc: {np.mean(eval_accs)}, std: {np.std(eval_accs)}")


if __name__ == "__main__":
    dataset_name = "PROTEINS"
    main(dataset_name)
