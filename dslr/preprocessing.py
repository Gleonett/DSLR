"""
Utils for data preprocessing
https://en.wikipedia.org/wiki/Feature_scaling
"""

import torch
import numpy as np
import pandas as pd
from torch import Tensor


class StandardScale(object):
    """
    This method is widely used for normalization in
    many machine learning algorithms
    x' = (x - mean(x)) / std(x)
    """
    mean: Tensor
    std: Tensor

    def fit(self, x: Tensor):
        """
        Obtain values for future scaling
        :param x: tensor of shape (num_samples, num_features)
        :return: None
        """
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0)

    def __call__(self, x: Tensor) -> Tensor:
        """
        Scale given set values
        :param x: tensor of shape (num_samples, num_features)
        :return: tensor of shape (num_samples, num_features)
        """
        return (x - self.mean) / self.std

    def to_dictionary(self) -> {str: Tensor}:
        """
        Put scale parameters to dictionary
        :return: None
        """
        return {"mean": self.mean, "std": self.std}

    def from_dictionary(self,
                        dictionary: {str: Tensor},
                        device: torch.device,
                        dtype: torch.dtype):
        """
        Load scale parameters from dictionary
        :return: None
        """
        self.mean = dictionary["mean"].to(device, dtype)
        self.std = dictionary["std"].to(device, dtype)


class MinMaxScale(object):
    """
    It is the simplest method and consists in rescaling the
    range of features to scale the range in [0, 1].
    x' = (x - min(x)) / (max(x) - min(x))
    """
    min: Tensor
    max: Tensor

    def fit(self, x: Tensor):
        """
        Obtain values for future scaling
        :param x: tensor of shape (num_samples, num_features)
        :return: None
        """
        self.min = x.min(dim=0).values
        self.max = x.max(dim=0).values

    def __call__(self, x: Tensor) -> Tensor:
        """
        Scale given set values
        :param x: tensor of shape (num_samples, num_features)
        :return: tensor of shape (num_samples, num_features)
        """
        return (x - self.min) / (self.max - self.min)

    def to_dictionary(self) -> {str: Tensor}:
        """
        Put scale parameters to dictionary
        :return: None
        """
        return {"min": self.min, "max": self.max}

    def from_dictionary(self,
                        dictionary: {str: Tensor},
                        device: torch.device,
                        dtype: torch.dtype):
        """
        Load scale parameters from dictionary
        :return: None
        """
        self.min = dictionary["min"].to(device, dtype)
        self.max = dictionary["max"].to(device, dtype)


scale = {
    "minmax": MinMaxScale(),
    "standard": StandardScale()
}


def fill_na(df: pd.DataFrame, courses: np.ndarray) -> pd.DataFrame:
    """
    This function split data to clusters and replace cluster nan values with
    calculated cluster mean value.
    "Birthday", "Best Hand" and course name are used for clustering.
    Clusters look like:
        2000 (Birthday) - "Right" (Best Hand) - course1 (course name)
                        |                     |_ course2
                        |                     |_ ...
                        |_"Left" - course1
                                 |_ ...
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
