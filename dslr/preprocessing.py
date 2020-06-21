import torch


class StandardScale(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, x):
        mean, std = [], []
        for i in range(0, x.shape[1]):
            mean.append(torch.mean(x[:, i]))
            std.append(torch.std(x[:, i]))
        self.mean = torch.stack(mean)
        self.std = torch.stack(std)

    def __call__(self, x):
        return (x - self.mean) / self.std


class MinMaxScale(object):
    def __init__(self, minimum=None, maximum=None):
        self.min = minimum
        self.max = maximum

    def fit(self, x):
        minimum, maximum = [], []
        for i in range(x.shape[1]):
            minimum.append(torch.min(x[:, i]))
            maximum.append(torch.max(x[:, i]))
        self.min = torch.stack(minimum)
        self.max = torch.stack(maximum)

    def __call__(self, x):
        return (x - self.min) / (self.max - self.min)
