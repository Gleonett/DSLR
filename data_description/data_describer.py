"""
Class for data description
"""

import numpy as np
import pandas as pd
from abc import ABC


class HogwartsDataDescriber(pd.DataFrame, ABC):

    houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
    colors = ['red', 'green', 'blue', 'yellow']

    @staticmethod
    def read_csv(csv_path: str):
        """
        Read .csv file
        :param csv_path: path to .csv file
        :return: HogwartsDataDescriber
        """
        return HogwartsDataDescriber(pd.read_csv(csv_path))

    def is_numeric(self, feature: str):
        """
        Check if column contains only numeric values
        :param feature: column name
        :return: Bool
        """
        return np.issubdtype(self[feature].dtype, np.number)

    def count(self, feature: str) -> int:
        """
        Number of the column elements without nans
        :param feature: column name
        :return: int
        """
        return len(self[feature].dropna())

    def mean(self, feature: str) -> float:
        """
        Mean value of the column elements
        :param feature: column name
        :return: float
        """
        return sum(self[feature].dropna()) / self.count(feature)

    def std(self, feature: str) -> float:
        """
        Compute the standard deviation, a measure of the spread
        of a distribution, of the column elements

        std = sqrt(mean(abs(x - x.mean())**2))
        :param feature: column name
        :return: float
        """
        dif = self[feature].dropna() - self.mean(feature)
        mean = sum(np.abs(dif) ** 2) / self.count(feature)
        return np.sqrt(mean)

    def min(self, feature: str) -> float:
        """
        Minimum value of the column elements
        :param feature: column name
        :return: float
        """
        tmp = np.nan
        for val in self[feature].dropna():
            tmp = tmp if val > tmp else val
        return tmp

    def max(self, feature: str) -> float:
        """
        Maximum value of the column elements
        :param feature: column name
        :return: float
        """
        tmp = -np.nan
        for val in self[feature].dropna():
            tmp = tmp if val < tmp else val
        return tmp

    def percentile(self, feature: str, percent: float) -> float:
        """
        Compute the percentile of the column elements
        :param feature: column name
        :param percent: value [0, 100]
        :return: float
        """
        arr = sorted(self[feature].dropna())
        k = (len(arr) - 1) * percent / 100
        f = np.floor(k)
        c = np.ceil(k)
        if f == c:
            return arr[int(k)]
        d0 = arr[int(f)] * (c - k)
        d1 = arr[int(c)] * (k - f)
        return d0 + d1
