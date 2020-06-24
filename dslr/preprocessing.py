import torch
import numpy as np
import pandas as pd


class StandardScale(object):

    mean: torch.Tensor
    std: torch.Tensor

    def fit(self, x):
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0)

    def __call__(self, x):
        return (x - self.mean) / self.std

    def to_dictionary(self):
        return {"mean": self.mean, "std": self.std}

    def from_dictionary(self, dictionary, device, dtype):
        self.mean = dictionary["mean"].to(device, dtype)
        self.std = dictionary["std"].to(device, dtype)


class MinMaxScale(object):

    min: torch.Tensor
    max: torch.Tensor

    def fit(self, x):
        self.min = x.min(dim=0).values
        self.max = x.max(dim=0).values

    def __call__(self, x):
        return (x - self.min) / (self.max - self.min)

    def to_dictionary(self):
        return {"min": self.min, "max": self.max}

    def from_dictionary(self, dictionary, device, dtype):
        self.min = dictionary["min"].to(device, dtype)
        self.max = dictionary["max"].to(device, dtype)


def fillna(df: pd.DataFrame, courses: np.ndarray):

    birthdays = df["Birthday"]
    int_birthdays = np.empty(birthdays.shape[0], dtype=np.int)
    for i, b in enumerate(birthdays):
        int_birthdays[i] = b.split('-')[0]
    df["Birthday"] = int_birthdays

    for b_key in np.unique(df["Birthday"]):
        for h_key in np.unique(df["Best Hand"]):
            for course in courses:
                mask = (df["Birthday"] == b_key) & (df["Best Hand"] == h_key)
                val = df.loc[mask, course].mean()
                df.loc[mask, course] = df.loc[mask, course].fillna(val)

    return df
