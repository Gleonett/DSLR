import torch
import numpy as np

from dslr.model import LogisticRegression


class OneVsAllLogisticRegression(LogisticRegression):

    def __init__(self,
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float,
                 transform=None,
                 lr=0.00001,
                 max_iterations=50000):

        super(OneVsAllLogisticRegression, self).__init__(device, dtype, transform, lr, max_iterations)
        self.device = device
        self.models = []
        self.labels = []

    def split_labels(self, y):
        self.labels = np.unique(y)
        splitted_labels = np.zeros((self.labels.shape[0], y.shape[0]))

        for label, new_y in zip(self.labels, splitted_labels):
            new_y[np.where(y == label)] = 1
        return splitted_labels

    def fit(self, x, y):
        if self.transform is not None:
            self.transform.fit(x)
            x = self.transform(x)

        splitted_labels = self.split_labels(y)
        for labels in splitted_labels:
            model = LogisticRegression(self.device, self.dtype, None, self.lr, self.max_iterations)
            model.fit(x, labels)
            self.models.append(model)

    def predict(self, x):
        if self.transform is not None:
            x = self.transform(x)

        p = []
        for model in self.models:
            p.append(model.predict(x))
        p = np.array(p).T
        p = np.argmax(p, axis=1)
        labels = []
        for prediction in p:
            labels.append(self.labels[prediction])
        return np.array(labels)
