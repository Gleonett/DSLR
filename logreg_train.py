"""
Script to train one-vs-all logistic regression
It saves models weights in weights.pt
"""

import numpy as np
import pandas as pd
from time import time
from argparse import ArgumentParser

from config import Config
from dslr.multi_classifier import OneVsAllLogisticRegression
from dslr.preprocessing import scale, fill_na


def train(data_path: str, weights_path: str, config_path: str):
    # CHOOSE FROM CONFIG FEATURES TO TRAIN ON
    config = Config(config_path)
    courses = config.choosed_features()

    preparation_t = time()
    df = pd.read_csv(data_path)
    df = fill_na(df, courses)

    x = df[courses].values
    y = df["Hogwarts House"].values

    model = OneVsAllLogisticRegression(
        device=config.device,
        transform=scale[config.scale],
        lr=config.lr,
        max_iterations=config.max_iterations
    )
    preparation_t = time() - preparation_t

    train_t = time()
    model.fit(x, y)
    train_t = time() - train_t

    model.save(weights_path)
    print("Preparation time:", np.round(preparation_t, 4))
    print("Training time:", np.round(train_t, 4))
    print("All time:", np.round(preparation_t + train_t, 4))


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('data_path', type=str,
                        help='Path to "dataset_train.csv" file')

    parser.add_argument('--weights_path', type=str, default="data/weights.pt",
                        help='Path to save weights file')

    parser.add_argument('--config_path', type=str, default="config.yaml",
                        help='path to .yaml file')

    args = parser.parse_args()

    train(args.data_path, args.weights_path, args.config_path)
