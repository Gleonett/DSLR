import torch
import numpy as np

from dslr.classifier import LogisticRegression
from dslr.pytorch_utils import get_device, to_tensor


class OneVsAllLogisticRegression(object):

    def __init__(self,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 transform=None,
                 lr=0.00001,
                 max_iterations=50000):

        self.device = get_device(device)
        self.dtype = dtype
        self.transform = transform
        self.lr = lr
        self.max_iterations = max_iterations
        self.models = []
        self.labels = None

    def predict(self, x):
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

    def fit(self, x, y):
        x = to_tensor(x, self.device, self.dtype)

        bin_labels = self._split_labels(y)
        bin_labels = to_tensor(bin_labels, self.device, self.dtype)

        if self.transform is not None:
            self.transform.fit(x)
            x = self.transform(x)

        for labels in bin_labels:
            model = LogisticRegression(self.device, self.dtype, None, self.lr, self.max_iterations)
            model.fit(x, labels)
            self.models.append(model)

    def _split_labels(self, y):
        self.labels = np.unique(y)
        splitted_labels = np.zeros((self.labels.shape[0], y.shape[0]))

        for label, new_y in zip(self.labels, splitted_labels):
            new_y[np.where(y == label)] = 1
        return splitted_labels
