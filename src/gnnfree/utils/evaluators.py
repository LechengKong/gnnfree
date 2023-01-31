import numpy as np
import torch
from abc import ABCMeta, abstractmethod


class Evaluator(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, name) -> None:
        self.name = name

    @abstractmethod
    def collect_res(self, res):
        pass

    @abstractmethod
    def summarize_res(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def better_results(self, a, b):
        pass

    @abstractmethod
    def init_result(self):
        pass


class MaxEvaluator(Evaluator):
    def better_results(self, a, b):
        return a > b

    def init_result(self):
        return -1e10


class MinEvaluator(Evaluator):
    def better_results(self, a, b):
        return a < b

    def init_result(self):
        return 1e10


class InfoNEEvaluator(MinEvaluator):
    def __init__(self, name) -> None:
        super().__init__(name)
        self.loss = []

    def collect_res(self, res):
        n = len(res)
        e_neg_mat = (
            res.view(-1)[1:].view(n - 1, n + 1)[:, :-1].reshape(n, n - 1)
        )
        e_pos = torch.diagonal(res)
        loss = -torch.mean(
            torch.log(torch.exp(e_pos) / torch.exp(e_neg_mat).sum(dim=-1))
        )
        self.loss.append(loss.item())

    def summarize_res(self):
        metrics = {}
        metrics["mi_loss"] = np.array(self.loss).mean()
        return metrics

    def reset(self):
        self.loss = []
