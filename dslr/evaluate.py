import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

from dslr.model import LogisticRegression

def train_test_split(x, y, test_size=0.3, random_state=None):
    if random_state:
        np.random.seed(random_state)

    y = y.T
    p = np.random.permutation(len(x))

    x_offset = int(len(x) * test_size)
    y_offset = int(len(y) * test_size)

    X_train = x[p][x_offset:]
    X_test = x[p][:x_offset]

    y_train = y[p][y_offset:]
    y_test = y[p][:y_offset]

    return X_train, X_test, y_train.T, y_test.T

class StandardScaler(object):
  def __init__(self, mean=np.array([]), std=np.array([])):
    self._mean = mean
    self._std = std

  def fit(self, X):
    for i in range(0, X.shape[1]):
      self._mean = np.append(self._mean, np.mean(X[:, i]))
      self._std = np.append(self._std, np.std(X[:, i]))

  def transform(self, X):
    return ((X - self._mean) / self._std)

def labels_split(y, houses):
    new_ys = np.zeros((len(houses), y.shape[0]))
    for house, new_y in zip(houses, new_ys):
        new_y[np.where(y == house)] = 1
    return new_ys


class MinMaxScaler(object):
  def __init__(self, min=np.array([]), std=np.array([])):
    self.min = min
    self.max = std

  def fit(self, X):
    for i in range(0, X.shape[1]):
      self.min = np.append(self.min, np.min(X[:, i]))
      self.max = np.append(self.max, np.max(X[:, i]))

  def transform(self, X):
    return (X - self.min) / (self.max - self.min)

def labels_split(y, houses):
    new_ys = np.zeros((len(houses), y.shape[0]))
    for house, new_y in zip(houses, new_ys):
        new_y[np.where(y == house)] = 1
    return new_ys


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--data_path',
                        type=str,
                        default='../data/dataset_train.csv',
                        help='Path to dataset_train.csv file')
    args = parser.parse_args()
    houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
    choosed_courses = ['Defense Against the Dark Arts', 'Muggle Studies', 'Divination', 'Herbology', 'Charms']

    df = pd.read_csv(args.data_path)
    for course in choosed_courses:
        df = df.dropna(subset=[course])
    x = np.array(df.values[:, [9, 17, 8, 10, 11]], dtype=float)
    y = df.values[:, 1]
    y = labels_split(y, houses)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=4)

    # sc = StandardScaler()
    sc = MinMaxScaler()
    sc.fit(X_train)

    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    models = []
    for bin_y in y_train:
        lr = LogisticRegression()
        lr.fit(X_train_std, bin_y)
        models.append(lr)

    p = []
    for model in models:
        p.append(model.predict(X_test_std))
    p = np.array(p).T

    y_pred = np.zeros(y_test.shape[0])
    y_pred = np.argmax(p, axis=1)
    y_test = np.argmax(y_test.T, axis=1)
    print(y_pred)
    print("Wrong predictions:", sum(y_test != y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
