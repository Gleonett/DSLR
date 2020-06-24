"""
Utils for data preprocessing
https://en.wikipedia.org/wiki/Feature_scaling
"""

import torch
import numpy as np
import pandas as pd


class StandardScale(object):
    """
    This method is widely used for normalization in many machine learning algorithms
    x' = (x - mean(x)) / std(x)
    """
    mean: torch.Tensor
    std: torch.Tensor

    def fit(self, x: torch.Tensor):
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0)

    def __call__(self, x: torch.Tensor):
        return (x - self.mean) / self.std

    def to_dictionary(self):
        return {"mean": self.mean, "std": self.std}

    def from_dictionary(self,
                        dictionary: dict,
                        device: torch.device,
                        dtype: torch.dtype):
        self.mean = dictionary["mean"].to(device, dtype)
        self.std = dictionary["std"].to(device, dtype)


class MinMaxScale(object):
    """
    It is the simplest method and consists in rescaling the
    range of features to scale the range in [0, 1].
    x' = (x - min(x)) / (max(x) - min(x))
    """
    min: torch.Tensor
    max: torch.Tensor

    def fit(self, x: torch.Tensor):
        self.min = x.min(dim=0).values
        self.max = x.max(dim=0).values

    def __call__(self, x: torch.Tensor):
        return (x - self.min) / (self.max - self.min)

    def to_dictionary(self):
        return {"min": self.min, "max": self.max}

    def from_dictionary(self,
                        dictionary: dict,
                        device: torch.device,
                        dtype: torch.dtype):
        self.min = dictionary["min"].to(device, dtype)
        self.max = dictionary["max"].to(device, dtype)


def fillna(df: pd.DataFrame, courses: np.ndarray):
    """
    This function split data to clusters and replace cluster's nan values with
    calculated cluster's mean value.
    "Birthday", "Best Hand" and course name are used for clustering.
    Clusters look like:
        2000 (Birthday) - "Right" (Best Hand) - course1 (course name)
                        \                     \_ course2
                         \                     \_ ...
                          \_"Left" - course1
                                   \_ ...
        1999 - ...

    :param df: dataframe to fill nan values
    :param courses: array of courses which will used for training/predicting
    :return: dataframe with filled nan values
    """

    # REPLACE FULL BIRTHDAY DATE WITH YEAR: "2000-03-30" -> 2000
    years = np.empty(df["Birthday"].shape[0], dtype=np.int)
    for i, b in enumerate(df["Birthday"]):
        years[i] = b.split('-')[0]
    df["Birthday"] = years

    # CLUSTERING
    for year in df["Birthday"].unique():
        for hand in df["Best Hand"].unique():
            for course in courses:
                # CHOOSE INDEXES WHICH HAVE CLUSTERS BIRTHDAY AND BEST HAND
                mask = (df["Birthday"] == year) & (df["Best Hand"] == hand)
                # CALCULATE MEAN IN CLUSTER
                val = df.loc[mask, course].mean()
                # FILL NAN VALUES WITH MEAN VALUE
                df.loc[mask, course] = df.loc[mask, course].fillna(val)
    return df
