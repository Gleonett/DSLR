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
        self.iteration = 0

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

        for self.iteration in range(self.max_iterations):

            # PERMUTATION FOR STOCHASTIC, MINI-BATCH OR BATCH GD
            perm = torch.randperm(x.shape[0])[:self.batch_size]

            # CALCULATE THE GRADIENT AT THE CURRENT POINT
            tmp_a, tmp_b = self._calculate_anti_gradient(x[perm], y[perm])

            # ADJUST LOGISTIC REGRESSION FREE MEMBER AND COEFFICIENTS
            self.a += self.lr * tmp_a / perm.shape[0]
            self.b += self.lr * tmp_b / perm.shape[0]

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
