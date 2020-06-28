"""
Script to train one-vs-all logistic regression
It saves models weights in weights.pt
"""

import numpy as np
import pandas as pd
from time import time
from argparse import ArgumentParser
from matplotlib import pyplot as plt

from config import Config
from dslr.preprocessing import scale, fill_na
from dslr.multi_classifier import OneVsAllLogisticRegression


def plot_training(model: OneVsAllLogisticRegression):
    """
    Plot loss history
    :param model: trained model
    :return: None
    """
    _, ax = plt.subplots()

    epochs = range(1, model.epochs + 1)

    for sub_model, label in zip(model.models, model.labels):
        ax.plot(epochs, sub_model.hist, label=label)

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Logistic Regression, batch size: {}'
                 .format(model.batch_size))
    ax.legend(loc="upper right")
    plt.show()


def train(data_path: str,
          weights_path: str,
          config_path: str,
          v: bool = False):

    # CHOOSE FROM CONFIG FEATURES TO TRAIN ON
    config = Config(config_path)
    courses = config.choosed_features()

    # READ TRAIN DATASET AND FILL NAN VALUES
    preparation_t = time()
    df = pd.read_csv(data_path)
    df = fill_na(df, courses)

    # CHOOSE FEATURE AND LABEL VALUES
    x = df[courses].values
    y = df["Hogwarts House"].values

    # CREATE MODEL TO TRAIN
    model = OneVsAllLogisticRegression(
        device=config.device,
        transform=scale[config.scale],
        lr=config.lr,
        epochs=config.epochs,
        batch_size=config.batch_size,
        seed=config.seed,
        save_hist=v
    )
    preparation_t = time() - preparation_t

    # TRAIN MODEL
    train_t = time()
    model.fit(x, y)
    train_t = time() - train_t

    # SAVE WEIGHTS AND SCALE PARAMS
    model.save(weights_path)

    print("Preparation time:", np.round(preparation_t, 4))
    print("Training time:", np.round(train_t, 4))
    print("All time:", np.round(preparation_t + train_t, 4))

    if v:
        plot_training(model)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('data_path', type=str,
                        help='Path to "dataset_train.csv" file')

    parser.add_argument('--weights_path', type=str, default="data/weights.pt",
                        help='Path to save weights file')

    parser.add_argument('--config_path', type=str, default="config.yaml",
                        help='path to .yaml file')

    parser.add_argument('-v', action="store_true",
                        help='visualize training')

    args = parser.parse_args()

    train(args.data_path, args.weights_path, args.config_path, args.v)
