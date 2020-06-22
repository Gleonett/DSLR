"""
Logistic regression implementation using PyTorch tensors
"""
import torch
import numpy as np

from dslr.pytorch_utils import get_device, to_tensor


class LogisticRegression(object):

    def __init__(self,
                 device: torch.device or str,
                 dtype: torch.dtype = torch.float,
                 transform=None,
                 lr=0.00001,
                 max_iterations=50000):
        """
        :param device: "cpu" or "cuda:{device_index}" usually device_index = 0
        :param dtype: type of data
        """
        if type(device) == str:
            self.device = get_device(device)
        else:
            self.device = device
        self.dtype = dtype
        self.transform = transform
        self.a = np.random.rand(1)[0] - 0.5
        self.b = None
        self.lr = lr
        self.max_iterations = max_iterations

    def predict(self, x):
        return 1.0 / (1.0 + torch.exp(x @ -self.b - self.a))

    def fit(self, x, y):
        if type(x) != torch.Tensor:
            x = to_tensor(x, self.device, self.dtype)
        if type(y) != torch.Tensor:
            y = to_tensor(y, self.device, self.dtype)

        if self.transform is not None:
            self.transform.fit(x)
            x = self.transform(x)

        self.b = torch.randn(x.shape[1]).uniform_(-0.5, 0.5)
        for i in range(self.max_iterations):
            tmp_a, tmp_b = self._calculate_anti_gradient(x, y)
            self.a = self.a + self.lr * tmp_a
            self.b = self.b + self.lr * tmp_b

    def _calculate_anti_gradient(self, x, y):
        p = self.predict(x)
        dif = y - p
        da = torch.sum(dif)
        db = x.t() @ dif
        return da, db

    def _to_tensor(self, x):
        return torch.from_numpy(x).to(self.device, self.dtype)
