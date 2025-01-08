from utils import BaseTrainer
from classifier.nn_classifier import NNClassifier

from torch import Tensor
from torch.nn.functional import cross_entropy


class NNClassifierTrainer(BaseTrainer[NNClassifier]):
    def loss(self, batch: list[Tensor]) -> Tensor:
        x, y = batch

        logits = self.model(x)
        loss = cross_entropy(logits, y)

        return loss
