import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

from pbmds import PBMDS, PBMDSTrainer, PBMDSDataset
from classifier import NNClassifier, NNClassifierTrainer


def train_pbmds(train_dmat_sqr: Tensor) -> PBMDS:
    dataset = PBMDSDataset(train_dmat_sqr)
    dataloader = DataLoader(dataset, batch_size=train_dmat_sqr.shape[0] // 2, shuffle=True)
    pbmds = PBMDS(train_dmat_sqr.shape[0], train_dmat_sqr.shape[0])
    trainer = PBMDSTrainer(
        pbmds,
        None,
        torch.optim.Adam(pbmds.parameters(), lr=1e-3),
    )
    trainer.train(dataloader, epochs=100, name=f"{dataset_name}_pbmds")
    return trainer.ema_model


def train_classifier(
    train_embeddings: Tensor,
    eval_embeddings: Tensor,
    train_labels: Tensor,
    eval_labels: Tensor,
) -> tuple[float, float]:
    classifier = NNClassifier(train_embeddings.shape[1], 2)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.8, step_size=10)
    trainer = NNClassifierTrainer(
        classifier,
        None,
        torch.optim.Adam(classifier.parameters(), lr=1e-2),
        scheduler,
    )
    dataloader = DataLoader(
        TensorDataset(train_embeddings, train_labels), batch_size=64, shuffle=True
    )
    eval_dataloader = DataLoader(
        TensorDataset(eval_embeddings, eval_labels), batch_size=64, shuffle=False
    )
    trainer.train(
        dataloader,
        epochs=10,
        name=f"{dataset_name}_classifier",
        eval_dataloader=eval_dataloader,
    )
    train_acc = trainer.eval(eval_dataloader=dataloader)
    eval_acc = trainer.eval(eval_dataloader=eval_dataloader)
    return train_acc, eval_acc


def main(dataset_name: str) -> None:
    data_file = np.load(f"data/dataset_name.npz")
    dmat_sqr = torch.tensor(data_file["dmat"]).float().square()
    labels = torch.tensor(data_file["labels"])

    kfold_validation = KFold(n_splits=10, shuffle=True)
    for i, (train_idx, test_idx) in enumerate(kfold_validation.split(np.arange(len(dmat_sqr))), 1): # type: ignore
        train_dmat_sqr = dmat_sqr[train_idx[:, None], train_idx]
        test_dmat_sqr = dmat_sqr[test_idx[:, None], train_idx]

        pbmds = train_pbmds(train_dmat_sqr)
        train_embeddings = pbmds.embedding(train_dmat_sqr)
        test_embeddings = pbmds.embedding(test_dmat_sqr)

        train_labels = labels[train_idx]
        eval_labels = labels[test_idx]

        train_acc, test_acc = train_classifier(
            train_embeddings, test_embeddings, train_labels, eval_labels
        )
        print(f"Fold {i}: Train acc: {train_acc}, Test acc: {test_acc}")


if __name__ == "__main__":
    dataset_name = "PROTEINS"
    main(dataset_name)
