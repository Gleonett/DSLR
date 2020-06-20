"""
Class for loading data
"""

import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from data_description.data_describer import HogwartsDataDescriber


class HogwartsDataset(Dataset):

    def __init__(self, csv_path: str, features: [str], transform=None):
        """

        :param csv_path:
        :param features:
        """
        self.data = HogwartsDataDescriber.read_csv(csv_path)

        for label in self.data.columns:
            if label not in features:
                self.data = self.data.drop(label, axis=1)

        self.transform = transform

    def __len__(self):
        return self.data.count("Index")

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = None
        ...

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--data_path',
                        type=str,
                        default='../data/dataset_train.csv',
                        help='Path to dataset_train.csv file')
    parser.add_argument('--course',
                        type=str,
                        default='Care of Magical Creatures',
                        help='Name of the course to plot')
    args = parser.parse_args()

    hd = HogwartsDataset(args.data_path, [args.course])
    print(hd.data)
