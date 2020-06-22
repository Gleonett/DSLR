

class StandardScale(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, x):
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0)

    def __call__(self, x):
        return (x - self.mean) / self.std


class MinMaxScale(object):
    def __init__(self, minimum=None, maximum=None):
        self.min = minimum
        self.max = maximum

    def fit(self, x):
        self.min = x.min(dim=0).values
        self.max = x.max(dim=0).values

    def __call__(self, x):
        return (x - self.min) / (self.max - self.min)
