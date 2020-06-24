"""
One-vs-all logistic regression implementation using PyTorch tensors
"""

import torch
import numpy as np

from dslr.classifier import LogisticRegression
from dslr.pytorch_utils import get_device, to_tensor


class OneVsAllLogisticRegression(object):

    def __init__(self,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 transform=None,
                 lr=0.001,
                 max_iterations=100):

        self.device = get_device(device)
        self.dtype = dtype
        self.transform = transform
        self.lr = lr
        self.max_iterations = max_iterations
        self.models = []
        self.labels = None

    def predict(self, x: torch.Tensor or np.ndarray):
        if type(x) != torch.Tensor:
            x = to_tensor(x, self.device, self.dtype)
        if self.transform is not None:
            x = self.transform(x)

        p = []
        for model in self.models:
            p.append(model.predict(x))
        p = torch.stack(p).t()
        p = torch.argmax(p, dim=1).cpu()
        labels = self.labels[p]
        return labels

    def fit(self, x: np.ndarray, y: np.ndarray):
        x = to_tensor(x, self.device, self.dtype)

        bin_labels = self._split_labels(y)
        bin_labels = to_tensor(bin_labels, self.device, self.dtype)

        if self.transform is not None:
            self.transform.fit(x)
            x = self.transform(x)

        for labels in bin_labels:
            model = LogisticRegression(self.device, self.dtype, self.lr, self.max_iterations)
            model.fit(x, labels)
            self.models.append(model)

    def _split_labels(self, y: np.ndarray):
        self.labels = np.unique(y)
        splitted_labels = np.zeros((self.labels.shape[0], y.shape[0]))

        for label, new_y in zip(self.labels, splitted_labels):
            new_y[np.where(y == label)] = 1
        return splitted_labels

    def save(self, path: str):
        models_w = {"transform": self.transform.to_dictionary()}
        for model, label in zip(self.models, self.labels):
            models_w[label] = model.to_dictionary()
        torch.save(models_w, path)

    def load(self, path: str):
        models_w = torch.load(path)

        self.transform.from_dictionary(models_w.pop("transform"), self.device, self.dtype)
        self.labels = np.array(list(models_w.keys()))

        for w in models_w.values():
            model = LogisticRegression(self.device, self.dtype, self.lr, self.max_iterations)
            model.from_dictionary(w)
            self.models.append(model)
