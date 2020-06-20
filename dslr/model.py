"""
Logistic regression implementation using PyTorch tensors
"""
import torch
import numpy as np


class LogisticRegression(object):

    def __init__(self, device: str = "cpu", dtype: torch.dtype = torch.float):
        """
        :param device: "cpu" or "cuda:{device_index}" usually device_index = 0
        :param dtype: type of data
        """
        self.device = self.get_device(device)
        self.dtype = dtype
        self.a = np.random.rand(1)[0] - 0.5
        self.b = None
        self.lr = 0.01
        self.max_iterations = 50

    @staticmethod
    def get_device(device):
        if "cpu" not in device:
            if torch.cuda.is_available():
                # TODO: test exception of wrong device index on machine with cuda
                device = torch.device(device)
            else:
                exit("Cuda not available")
        else:
            device = torch.device(device)
        return device

    def predict(self, x):
        return 1.0 / (1.0 + np.exp(-x.dot(self.b) - self.a))

    def fit(self, x, y):
        self.b = np.random.rand(x.shape[1]) - 0.5
        for i in range(self.max_iterations):
            tmp_a, tmp_b = self._calculate_anti_gradient(x, y)
            self.a = self.a + self.lr * tmp_a
            self.b = self.b + self.lr * tmp_b

    def _calculate_anti_gradient(self, x, y):
        p = self.predict(x)
        da = np.sum(y - p)
        db = x.transpose().dot(y - p)
        return da, db


if __name__ == "__main__":
    model = LogisticRegression(device="cpu")
    print(model.device)
