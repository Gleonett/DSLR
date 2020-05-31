import numpy as np
import pandas as pd
from abc import ABC


class HogwartsDataDescriber(pd.DataFrame, ABC):

    @staticmethod
    def read_csv(csv_path):
        return HogwartsDataDescriber(pd.read_csv(csv_path))

    def is_numeric(self, feature: str):
        return np.issubdtype(self[feature].dtype, np.number)

    def count(self, feature: str):
        return len(self[feature].dropna())

    def mean(self, feature: str):
        return sum(self[feature].dropna()) / self.count(feature)

    def std(self, feature: str):
        dif = self[feature].dropna() - self.mean(feature)
        mean = sum(np.abs(dif) ** 2) / self.count(feature)
        return np.sqrt(mean)

    def min(self, feature: str):
        tmp = np.nan
        for val in self[feature].dropna():
            tmp = tmp if val > tmp else val
        return tmp

    def max(self, feature: str):
        tmp = -np.nan
        for val in self[feature].dropna():
            tmp = tmp if val < tmp else val
        return tmp

    def percentile(self, feature: str, percent: float):
        arr = sorted(self[feature].dropna())
        k = (len(arr) - 1) * percent / 100
        f = np.floor(k)
        c = np.ceil(k)
        if f == c:
            return arr[int(k)]
        d0 = arr[int(f)] * (c - k)
        d1 = arr[int(c)] * (k - f)
        return d0 + d1
