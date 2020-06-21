import numpy as np


class StandardScale(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, x):
        mean, std = [], []
        for i in range(0, x.shape[1]):
            mean.append(np.mean(x[:, i]))
            std.append(np.std(x[:, i]))
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, x):
        return (x - self.mean) / self.std


class MinMaxScale(object):
    def __init__(self, minimum=None, maximum=None):
        self.min = minimum
        self.max = maximum

    def fit(self, x):
        minimum, maximum = [], []
        for i in range(x.shape[1]):
            minimum.append(np.min(x[:, i]))
            maximum.append(np.max(x[:, i]))
        self.min = np.array(minimum)
        self.max = np.array(maximum)

    def __call__(self, x):
        return (x - self.min) / (self.max - self.min)
