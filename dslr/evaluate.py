import numpy as np
import pandas as pd
from time import time

from sklearn.metrics import accuracy_score

from dslr.multi_classifier import OneVsAllLogisticRegression
from dslr.preprocessing import MinMaxScale, StandardScale


def train_test_split(x, y, test_size=0.3, random_state=None):
    if random_state:
        np.random.seed(random_state)

    p = np.random.permutation(len(x))

    x_offset = int(len(x) * test_size)
    y_offset = int(len(y) * test_size)

    X_train = x[p][x_offset:]
    X_test = x[p][:x_offset]

    y_train = y[p][y_offset:]
    y_test = y[p][:y_offset]

    return X_train, X_test, y_train, y_test


def evaluate(data_path):
    # houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
    choosed_courses = ['Ancient Runes', 'Divination', 'Herbology', 'Charms', 'Flying']

    df = pd.read_csv(data_path)
    for course in choosed_courses:
        df = df.dropna(subset=[course])
        # df[course].fillna(df[course].median())

    x = np.array(df[choosed_courses].values, dtype=float)
    y = df.values[:, 1]

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=5)

    model = OneVsAllLogisticRegression(
        transform=MinMaxScale(),
        lr=0.00001,
        max_iterations=50000
    )
    model.fit(X_train, y_train)
    p = model.predict(X_test)

    print("Wrong predictions:", sum(y_test != p))
    print("Accuracy:", np.round(accuracy_score(y_test, p), 4))


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--data_path',
                        type=str,
                        default='../data/dataset_train.csv',
                        help='Path to dataset_train.csv file')
    args = parser.parse_args()
    evaluate(args.data_path)
