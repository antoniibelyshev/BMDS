import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from pbmds import VBE, VBETrainer, VBEDataset


def train_vbe(
    train_dmat_sqr: Tensor,
    dataset_name: str,
    *,
    lr: float = 4e-4,
    epochs: int = 100,
) -> VBE:
    torch.manual_seed(0)
    dataset = VBEDataset(train_dmat_sqr)
    dataloader = DataLoader(dataset, batch_size=train_dmat_sqr.shape[0], shuffle=True, drop_last=True)
    vbe = VBE(train_dmat_sqr.shape[0], train_dmat_sqr.shape[0])
    optimizer = torch.optim.Adam(vbe.parameters(), lr=lr)
    scheduler = None
    trainer = VBETrainer(
        vbe,
        None,
        optimizer,
        scheduler,
    )
    trainer.train(dataloader, epochs=epochs, name=f"{dataset_name}_vbe")
    return trainer.ema_model


def main(dataset_name: str) -> None:
    data_file = np.load(f"data/{dataset_name}.npz")
    dmat_sqr = torch.tensor(data_file["pw_dmat"]).float().square()
    labels = torch.tensor(data_file["labels"])

    kfold_validation = KFold(n_splits=10, shuffle=True, random_state=0)
    for i, (train_idx, eval_idx) in enumerate(kfold_validation.split(np.arange(len(dmat_sqr))), 1): # type: ignore
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_dmat_sqr = dmat_sqr[train_idx][:, train_idx].to(device)
        eval_dmat_sqr = dmat_sqr[eval_idx][:, train_idx].to(device)

        vbe = train_vbe(train_dmat_sqr, dataset_name)
        x_train = vbe.embedding(train_dmat_sqr)
        x_eval = vbe.embedding(eval_dmat_sqr)

        y_train = labels[train_idx]
        y_eval = labels[eval_idx]

        np.savez(
            f"tmp/{dataset_name}_fold_{i}_embeddings.npz",
            train_dmat_sqr = train_dmat_sqr.cpu(),
            eval_dmat_sqr = eval_dmat_sqr.cpu(),
            x_train = x_train,
            x_eval = x_eval,
            y_train = y_train,
            y_eval = y_eval,
        )

        torch.save(vbe.state_dict(), f"tmp/{dataset_name}_fold_{i}_vbe.pt")


if __name__ == "__main__":
    main("PROTEINS")
