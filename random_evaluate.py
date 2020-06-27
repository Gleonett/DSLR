"""
Script to train and evaluate one-vs-all logistic regression
on dataset_train.csv
"""

import numpy as np
import pandas as pd
from time import time
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score

from config import Config
from dslr.preprocessing import scale, fill_na
from dslr.multi_classifier import OneVsAllLogisticRegression


def train_test_split(x: np.ndarray,
                     y: np.ndarray,
                     test_part=0.3,
                     random_state: int or None = None):
    np.random.seed(random_state)

    p = np.random.permutation(len(x))

    x_offset = int(len(x) * test_part)
    y_offset = int(len(y) * test_part)

    x_train = x[p][x_offset:]
    x_test = x[p][:x_offset]

    y_train = y[p][y_offset:]
    y_test = y[p][:y_offset]

    return x_train, x_test, y_train, y_test


def evaluate(data_path: str,
             config_path: str,
             test_part: float):
    # CHOOSE FROM CONFIG FEATURES TO TRAIN AND PREDICT
    config = Config(config_path)
    courses = config.choosed_features()

    preparation_t = time()
    df = pd.read_csv(data_path)
    df = fill_na(df, courses)

    x = df[courses].values
    y = df["Hogwarts House"].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_part, config.seed)

    model = OneVsAllLogisticRegression(
        device=config.device,
        transform=scale[config.scale],
        lr=config.lr,
        max_iterations=config.max_iterations,
        batch_size=config.batch_size,
        seed=config.seed
    )
    preparation_t = time() - preparation_t

    train_t = time()
    model.fit(x_train, y_train)
    train_t = time() - train_t

    predict_t = time()
    p = model.predict(x_test)
    predict_t = time() - predict_t

    print("Wrong predictions:", sum(y_test != p))
    print("Accuracy:", np.round(accuracy_score(y_test, p), 4), end='\n\n')
    print("Preparation time:", np.round(preparation_t, 4))
    print("Training time:", np.round(train_t, 4))
    print("Prediction time:", np.round(predict_t, 4))
    print("All time:", np.round(preparation_t + train_t + predict_t, 4))


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('--data_path',
                        type=str,
                        default='data/dataset_train.csv',
                        help='Path to dataset_train.csv file')

    parser.add_argument('--config_path',
                        type=str,
                        default='config.yaml',
                        help='path to .yaml file')

    parser.add_argument('--test_part',
                        type=float,
                        default=0.3,
                        help='Percent of test part. "0.3" means model will '
                             'train on 0.7 of data and evaluate at other 0.3')

    args = parser.parse_args()

    evaluate(args.data_path, args.config_path, args.test_part)
