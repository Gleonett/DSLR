import numpy as np
import pandas as pd
from time import time
from argparse import ArgumentParser

from config import Config
from dslr.preprocessing import MinMaxScale, StandardScale
from dslr.multi_classifier import OneVsAllLogisticRegression


scale = {
    "minmax": MinMaxScale(),
    "standard": StandardScale()
}


def train(data_path, config_path):
    config = Config(config_path)
    courses = np.array(list(config.features.keys()))
    mask = np.array(list(config.features.values()))
    choosed_courses = courses[mask]

    preparation_t = time()
    df = pd.read_csv(data_path)
    for course in choosed_courses:
        df = df.dropna(subset=[course])

    x = np.array(df[choosed_courses].values, dtype=float)
    y = df.values[:, 1]
    preparation_t = time() - preparation_t

    train_t = time()
    model = OneVsAllLogisticRegression(
        device=config.device,
        transform=scale[config.scale],
        lr=config.lr,
        max_iterations=config.max_iterations
    )
    model.fit(x, y)
    train_t = time() - train_t

    print("Preparation time:", np.round(preparation_t, 4))
    print("Training time:", np.round(train_t, 4))
    print("All time:", np.round(preparation_t + train_t, 4))


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('data_path', type=str,
                        help='Path to "dataset_train.csv" file')

    parser.add_argument('--config_path', type=str, default="config.yaml",
                        help='path to .yaml file')

    args = parser.parse_args()
    train(args.data_path, args.config_path)
