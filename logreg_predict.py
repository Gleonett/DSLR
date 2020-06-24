"""
Script to predict labels with one-vs-all logistic regression.
It save predicted labels in houses.csv
"""

import os
import numpy as np
import pandas as pd
from time import time
from argparse import ArgumentParser

from config import Config
from dslr.preprocessing import scale, fill_na
from dslr.multi_classifier import OneVsAllLogisticRegression


def predict(data_path: str, weights_path: str, output_folder: str, config_path: str):
    config = Config(config_path)
    courses = np.array(list(config.features.keys()))
    mask = np.array(list(config.features.values()))
    choosed_courses = courses[mask]

    preparation_t = time()
    df = pd.read_csv(data_path)
    df = fill_na(df, choosed_courses)

    x = df[choosed_courses].values

    model = OneVsAllLogisticRegression(
        device=config.device,
        transform=scale[config.scale],
        lr=config.lr,
        max_iterations=config.max_iterations
    )

    model.load(weights_path)

    preparation_t = time() - preparation_t

    predict_t = time()
    p = model.predict(x)
    predict_t = time() - predict_t

    pred = pd.DataFrame(p, columns=["Hogwarts House"])
    pred.to_csv(os.path.join(output_folder, "houses.csv"),
                index_label="Index")

    print("Preparation time:", np.round(preparation_t, 4))
    print("Prediction time:", np.round(predict_t, 4))
    print("All time:", np.round(preparation_t + predict_t, 4))


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('data_path', type=str,
                        help='Path to "dataset_test.csv" file')

    parser.add_argument('weights_path', type=str,
                        help='Path to "weights.pt" file')

    parser.add_argument('--output_folder', type=str, default="data",
                        help='Path to folder where to save houses.csv')

    parser.add_argument('--config_path', type=str, default="config.yaml",
                        help='Path to .yaml file')

    args = parser.parse_args()
    predict(args.data_path, args.weights_path, args.output_folder, args.config_path)
