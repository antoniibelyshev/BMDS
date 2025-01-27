from utils import BaseTrainer, safe_log
from .pbmds import PBMDS

from torch.nn.functional import relu
from torch import Tensor
import wandb
from typing import Any


class PBMDSTrainer(BaseTrainer[PBMDS]):    
    def loss(self, batch: list[Tensor]) -> Tensor:
        x1, x2, s = batch

        dist_sqr = (self.model(x1) - self.model(x2)).square().sum(1)
        # dist_sqr = relu(dist_sqr - 1e-8) + 1e-8 # for numerical stability
        ratio = s / dist_sqr
        loss = (ratio - safe_log(ratio) - 1).mean()
        reg = self.model.regularization()
        wandb.log({"reg": reg.item()})
        return loss + reg

    def eval(self, **kwargs: Any) -> None:
        wandb.log({"relevant dims count": self.ema_model.relevant_dims().float().sum()})
