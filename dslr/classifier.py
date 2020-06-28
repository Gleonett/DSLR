"""
Logistic regression implementation using PyTorch tensors

https://en.wikipedia.org/wiki/Logistic_regression
"""

import torch
from torch import Tensor


class LogisticRegression(object):
    a: Tensor
    b: Tensor

    def __init__(self, device: torch.device,
                 dtype: torch.dtype,
                 batch_size: int,
                 epochs: int = 100,
                 lr: float = 0.001, save_hist: bool = False):
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.iteration = 0
        self.hist = [] if save_hist else None

    def predict(self, x: Tensor) -> Tensor:
        """
        Calculate the probability of assigning input objects
        to the first class.
        :param x: tensor of shape (num_samples, num_features)
        :return: tensor of shape (num_samples)
        """
        return 1.0 / (1.0 + torch.exp(x @ -self.b - self.a))

    def fit(self, x: Tensor, y: Tensor):
        """
        Train logistic regression on a given training set using
        the gradient method
        :param x: tensor of shape (num_samples, num_features)
        :param y: tensor of shape (num_samples) - labels
        :return: None
        """

        # INIT FREE MEMBER AND COEFFICIENTS
        self.a = torch.randn(1).uniform_(-0.5, 0.5).to(self.device)
        self.b = torch.randn(x.shape[1]).uniform_(-0.5, 0.5).to(self.device)

        for self.iteration in range(self.epochs):

            # PERMUTATION FOR STOCHASTIC, MINI-BATCH OR BATCH GD
            perm = torch.randperm(x.shape[0])[:self.batch_size]

            # CALCULATE THE GRADIENT AT THE CURRENT POINT
            tmp_a, tmp_b = self._calculate_anti_gradient(x[perm], y[perm])

            # ADJUST LOGISTIC REGRESSION FREE MEMBER AND COEFFICIENTS
            self.a += self.lr * tmp_a / perm.shape[0]
            self.b += self.lr * tmp_b / perm.shape[0]

            # SAVE HISTORY FOR FUTURE VISUALIZATION
            if self.hist is not None:
                self.hist.append(self._loss(x, y))

    def _calculate_anti_gradient(self,
                                 x: Tensor,
                                 y: Tensor) -> (Tensor, Tensor):
        """
        Calculate anti gradient of logarithm of the likelihood function
        on a given training set
        :param x: tensor of shape (num_samples, num_features)
        :param y: tensor of shape (num_samples) - labels
        :return: (tensor of shape (1), tensor of shape (num_features))
        """
        p = self.predict(x)
        dif = y - p

        # CALCULATE FREE MEMBER
        da = torch.sum(dif)

        # CALCULATE COEFFICIENTS
        db = x.t() @ dif
        return da, db

    def _loss(self, x: Tensor, y: Tensor) -> float:
        """
        Calculate the logarithm of the likelihood function
        on a given training set
        :param x: tensor of shape (num_samples, num_features)
        :param y: tensor of shape (num_samples) - labels
        :return: logarithm of the likelihood function
        """
        # TO PREVENT LOGARITHM OF ZERO
        p = self.predict(x) + 0.000001

        loss = torch.sum(y * torch.log(p) +
                         (1.0 - y) * torch.log(1.0 - p)) / -x.shape[0]
        return float(loss.cpu().numpy())

    def to_dictionary(self) -> {str: Tensor}:
        """
        Put model parameters to dictionary
        :return: None
        """
        return {"a": self.a, "b": self.b}

    def from_dictionary(self, dictionary: {str: Tensor}):
        """
        Load model parameters from dictionary
        :return: None
        """
        self.a = dictionary["a"].to(self.device, self.dtype)
        self.b = dictionary["b"].to(self.device, self.dtype)
