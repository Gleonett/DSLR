import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import accuracy_score

from config import Config
from dslr.preprocessing import MinMaxScale, StandardScale
from dslr.multi_classifier import OneVsAllLogisticRegression


scale = {
    "minmax": MinMaxScale(),
    "standard": StandardScale()
}


def train_test_split(x, y, test_size=0.3, random_state=None):
    if random_state:
        np.random.seed(random_state)

    p = np.random.permutation(len(x))

    x_offset = int(len(x) * test_size)
    y_offset = int(len(y) * test_size)

    x_train = x[p][x_offset:]
    x_test = x[p][:x_offset]

    y_train = y[p][y_offset:]
    y_test = y[p][:y_offset]

    return x_train, x_test, y_train, y_test


def evaluate(data_path, config_path, test_part, state):
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

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_part, random_state=state)

    model = OneVsAllLogisticRegression(
        device=config.device,
        transform=scale[config.scale],
        lr=config.lr,
        max_iterations=config.max_iterations
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
    from argparse import ArgumentParser

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
                        help='Percent of test part. "0.3" means model will'
                             ' train on 0.7 of data and evaluate at other 0.3')

    parser.add_argument('--state',
                        type=int or None,
                        default=5,
                        help='Random state to reproduce results.'
                             '"-1" means no random state')

    args = parser.parse_args()

    if args.state == -1:
        args.state = None

    evaluate(args.data_path, args.config_path, args.test_part, args.state)
