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
                 transform: callable or None = None,
                 lr: float = 0.001,
                 epochs: int = 100,
                 batch_size: int or None = None,
                 seed: int or None = None,
                 save_hist: bool = False):

        self.device = get_device(device)
        self.dtype = dtype
        self.transform = transform
        self.lr = lr
        self.epochs = epochs
        self.models = []
        self.labels = None
        self.batch_size = batch_size
        self.save_hist = save_hist
        if type(seed) == int:
            torch.manual_seed(seed)

    def predict(self, x: torch.Tensor or np.ndarray) -> np.ndarray:
        """
        Predict labels for given set
        :param x: tensor or array of shape (num_samples, num_features)
        :return: array of labels of shape (num_samples)
        """
        if type(x) != torch.Tensor:
            x = to_tensor(x, self.device, self.dtype)

        # SCALE GIVEN FEATURES
        if self.transform is not None:
            x = self.transform(x)

        # CALCULATE THE PROBABILITY OF ASSIGNING INPUT OBJECTS TO EACH CLASS
        p = []
        for model in self.models:
            p.append(model.predict(x))

        # LABELING ACCORDING TO BEST PREDICTION
        p = torch.stack(p).t()
        p = torch.argmax(p, dim=1).cpu()
        labels = self.labels[p]
        return labels

    def fit(self, x: np.ndarray, y: np.ndarray):
        """
        Train multiple logistic regression models on given training set.
        1 model per label
        :param x: tensor of shape (num_samples, num_features)
        :param y: array with labels of shape (num_samples)
        :return: None
        """
        if self.batch_size is None:
            self.batch_size = x.shape[0]

        x = to_tensor(x, self.device, self.dtype)

        # SPLIT LABELS INTO ONE-VS-ALL SETS
        bin_labels = self._split_labels(y)
        bin_labels = to_tensor(bin_labels, self.device, self.dtype)

        # SCALE GIVEN FEATURES
        if self.transform is not None:
            self.transform.fit(x)
            x = self.transform(x)

        for labels in bin_labels:
            # CREATE MODEL FOR CLASSIFICATION OF CURRENT LABEL
            model = LogisticRegression(self.device,
                                       self.dtype,
                                       self.batch_size,
                                       self.epochs,
                                       self.lr,
                                       self.save_hist)
            # TRAIN MODEL
            model.fit(x, labels)
            self.models.append(model)

    def _split_labels(self, y: np.ndarray) -> np.ndarray:
        """
        Split labels into one-vs-all sets and binarize them
        :param y: array with labels of shape (num_samples)
        :return: array of shape (num_unique_labels, num_samples)
                with 0, 1 values
        """
        self.labels = np.unique(y)
        splitted_labels = np.zeros((self.labels.shape[0], y.shape[0]))

        for label, new_y in zip(self.labels, splitted_labels):
            new_y[np.where(y == label)] = 1
        return splitted_labels

    def save(self, path: str):
        """
        Save model parameters
        :param path: path where to save weights
        :return: None
        """
        models_w = {"transform": self.transform.to_dictionary()}
        for model, label in zip(self.models, self.labels):
            models_w[label] = model.to_dictionary()
        torch.save(models_w, path)

    def load(self, path: str):
        """
        Load model parameters from .pt file
        :param path: path to .pt file
        :return: None
        """
        models_w = torch.load(path)

        self.transform.from_dictionary(models_w.pop("transform"),
                                       self.device,
                                       self.dtype)
        self.labels = np.array(list(models_w.keys()))

        for w in models_w.values():
            model = LogisticRegression(self.device,
                                       self.dtype,
                                       self.batch_size,
                                       self.epochs,
                                       self.lr)
            model.from_dictionary(w)
            self.models.append(model)
