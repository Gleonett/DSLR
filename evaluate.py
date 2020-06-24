import os
import pandas as pd
from argparse import ArgumentParser
from sklearn.metrics import accuracy_score

from logreg_train import train
from logreg_predict import predict


def evaluate(train_path,
             test_path,
             truth_path,
             weights_path,
             output_folder,
             config_path):

    print("Training:")
    train(train_path, config_path)
    print('+' * 30)

    print("Predicting:")
    predict(test_path, weights_path, output_folder, config_path)
    print('-' * 30)

    pred = pd.read_csv(os.path.join(output_folder, "houses.csv"))
    true = pd.read_csv(truth_path)

    y_pred = pred['Hogwarts House']
    y_true = true['Hogwarts House']
    print("Wrong predictions:", sum(y_true != y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--train_path', type=str, default="data/dataset_train.csv",
                        help='Path to "dataset_train.csv" file')

    parser.add_argument('--test_path', type=str, default="data/dataset_test.csv",
                        help='Path to "dataset_test.csv" file')

    parser.add_argument('--truth_path', type=str, default="data/dataset_truth.csv",
                        help='Path to "dataset_truth.csv" file')

    parser.add_argument('--weights_path', type=str, default="data/weights.pt",
                        help='Path to "weights.pt" file')

    parser.add_argument('--output_folder', type=str, default="data",
                        help='Path to folder where to save houses.csv')

    parser.add_argument('--config_path', type=str, default="config.yaml",
                        help='Path to .yaml file')

    args = parser.parse_args()

    evaluate(args.train_path,
             args.test_path,
             args.truth_path,
             args.weights_path,
             args.output_folder,
             args.config_path)
