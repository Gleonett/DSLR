"""
Logistic regression implementation using PyTorch tensors

https://en.wikipedia.org/wiki/Logistic_regression
"""

import torch
from torch import Tensor


class LogisticRegression(object):

    a: Tensor
    b: Tensor

    def __init__(self,
                 device: torch.device,
                 dtype: torch.dtype,
                 lr: float = 0.001,
                 max_iterations: int = 100,
                 batch_size: int or None = None):
        self.device = device
        self.dtype = dtype
        self.lr = lr
        self.max_iterations = max_iterations
        self.batch_size = batch_size

    def predict(self, x: Tensor):
        return 1.0 / (1.0 + torch.exp(x @ -self.b - self.a))

    def fit(self, x: Tensor, y: Tensor):

        self.a = torch.randn(1).uniform_(-0.5, 0.5).to(self.device)
        self.b = torch.randn(x.shape[1]).uniform_(-0.5, 0.5).to(self.device)

        for i in range(self.max_iterations):
            perm = torch.randperm(x.shape[0])[:self.batch_size]
            tmp_a, tmp_b = self._calculate_anti_gradient(x[perm], y[perm])
            self.a = self.a + self.lr * tmp_a
            self.b = self.b + self.lr * tmp_b

    def _calculate_anti_gradient(self, x: Tensor, y: Tensor):
        p = self.predict(x)
        dif = y - p
        da = torch.sum(dif)
        db = x.t() @ dif
        return da, db

    def to_dictionary(self):
        return {"a": self.a, "b": self.b}

    def from_dictionary(self, dictionary: {str: Tensor}):
        self.a = dictionary["a"].to(self.device, self.dtype)
        self.b = dictionary["b"].to(self.device, self.dtype)
