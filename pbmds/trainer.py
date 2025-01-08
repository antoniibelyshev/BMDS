from utils import BaseTrainer, safe_log
from .pbmds import PBMDS

# from torch.nn.functional import relu

from torch import Tensor


class PBMDSTrainer(BaseTrainer[PBMDS]):    
    def loss(self, batch: list[Tensor]) -> Tensor:
        x1, x2, s = batch

        dist_sqr = (self.model(x1) - self.model(x2)).pow(2).sum(1)
        # dist_sqr = relu(dist_sqr - 1e-8) + 1e-8 # for numerical stability
        ratio = s / dist_sqr
        loss = (ratio - safe_log(ratio) - 1).mean()

        return loss
