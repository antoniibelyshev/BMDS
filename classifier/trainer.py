from utils import BaseTrainer
from classifier.nn_classifier import NNClassifier

from torch import Tensor
from torch.nn.functional import cross_entropy
from torch.utils.data import DataLoader
import wandb

from typing import Any


class NNClassifierTrainer(BaseTrainer[NNClassifier]):
    def loss(self, batch: list[Tensor]) -> Tensor:
        x, y = batch

        logits = self.model(x.to(self.device))
        loss = cross_entropy(logits, y.to(self.device))

        return loss
    
    def eval(self, *, eval_dataloader: DataLoader[tuple[Tensor, ...]], **kwargs: Any) -> float:
        eval_loss = 0
        eval_acc = 0
        count = 0
        for x, y in eval_dataloader:
            logits = self.ema_model(x.to(self.device))
            loss = cross_entropy(logits, y.to(self.device))
            eval_loss += loss.item() * x.shape[0]
            eval_acc += (logits.argmax(1) == y.to(self.device)).sum().item()
            count += x.shape[0]
        try:
            wandb.log({"eval_loss": eval_loss / count, "eval_acc": eval_acc / count})
        except:
            pass
        return eval_acc / count
