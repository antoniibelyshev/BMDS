from typing import Tuple, Dict, Any, Optional

import pytorch_lightning.loggers
import torch
import wandb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold

from .bmds import SklearnBMDS
from .classifier import SklearnMLPClassifier


def train_eval_bmds(
        name: str,
        dist_mat_train: torch.Tensor,
        dist_mat_eval: torch.Tensor,
        *,
        bmds_kwargs: Optional[Dict[str, Any]] = None,
        bmds_train_kwargs: Optional[Dict[str, Any]] = None,
        bmds_eval_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bmds = SklearnBMDS(**(bmds_kwargs or dict()))

    run = wandb.init(project=name + " BMDS train")
    logger = pytorch_lightning.loggers.WandbLogger()
    x_train, std_train = bmds.fit_transform(dist_mat_train, logger=logger, **(bmds_train_kwargs or dict()))
    run.finish()

    run = wandb.init(project=name + " BMDS eval")
    logger = pytorch_lightning.loggers.WandbLogger()
    x_eval, std_eval = bmds.transform(dist_mat_eval, logger=logger, **(bmds_eval_kwargs or dict()))
    run.finish()

    return x_train, std_train, x_eval, std_eval


def train_eval_clf(
        name: str,
        x_train: torch.Tensor,
        y_train: torch.Tensor,
        x_eval: torch.Tensor,
        y_eval: torch.Tensor,
        std_train: Optional[torch.Tensor] = None,
        std_eval: Optional[torch.Tensor] = None,
        *,
        clf_kwargs: Optional[Dict[str, Any]] = None,
        clf_train_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    clf = SklearnMLPClassifier(y_train.max() + 1, **(clf_kwargs or dict()))

    run = wandb.init(project=name + " clf")
    logger = pytorch_lightning.loggers.WandbLogger()
    pred_train = clf.fit_predict(x_train, y_train, std_train, logger=logger, **(clf_train_kwargs or dict()))
    pred_eval_rand = clf.predict(x_eval, std_eval)
    pred_eval = clf.predict(x_eval)
    logger.log_hyperparams({
        "train_acc": accuracy_score(pred_train, y_train),
        "eval_acc_rand": accuracy_score(pred_eval_rand, y_eval),
        "eval_acc": accuracy_score(pred_eval, y_eval),
    })
    run.finish()


def train_eval(
        name: str,
        dist_mat: torch.Tensor,
        y: torch.Tensor,
        idx_train_eval: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *,
        bmds_kwargs: Optional[Dict[str, Any]] = None,
        bmds_train_kwargs: Optional[Dict[str, Any]] = None,
        bmds_eval_kwargs: Optional[Dict[str, Any]] = None,
        clf_kwargs: Optional[Dict[str, Any]] = None,
        clf_train_kwargs: Optional[Dict[str, Any]] = None,
):
    if idx_train_eval is None:
        idx_train, idx_eval = train_test_split(torch.arange(len(y)))
    else:
        idx_train, idx_eval = idx_train_eval

    dist_mat_train = dist_mat[idx_train[:, None], idx_train]
    y_train = y[idx_train]
    dist_mat_eval = dist_mat[idx_train[:, None], idx_eval]
    y_eval = y[idx_eval]

    x_train, std_train, x_eval, std_eval = train_eval_bmds(
        name, dist_mat_train, dist_mat_eval,
        bmds_kwargs=bmds_kwargs, bmds_train_kwargs=bmds_train_kwargs, bmds_eval_kwargs=bmds_eval_kwargs,
    )

    train_eval_clf(
        name, x_train, y_train, x_eval, y_eval, std_train, std_eval,
        clf_kwargs=clf_kwargs, clf_train_kwargs=clf_train_kwargs,
    )


def train_eval_k_fold(
        name: str,
        dist_mat: torch.Tensor,
        y: torch.Tensor,
        *,
        k: int = 5,
        **kwargs: Dict[str, Any],
) -> None:
    name += " k-fold"
    k_fold = KFold(n_splits=k, shuffle=True, random_state=42)
    for idx_train_eval in k_fold.split(torch.arange(len(y))):
        train_eval(name, dist_mat, y, idx_train_eval, **kwargs)
