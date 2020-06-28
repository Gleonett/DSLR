"""
3D plot of clusters mean values
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.append("..")
from config import Config
from data_description.describe import abbreviation


def clusters_3d(df: pd.DataFrame, courses: np.ndarray):
    """
    This function split data to clusters and visualise clusters mean values.
    "Birthday", "Best Hand" and course name are used for clustering.
    Clusters look like:
        2000 (Birthday) - "Right" (Best Hand) - course1 (course name)
                        |                     |_ course2
                        |                     |_ ...
                        |_"Left" - course1
                                  |_ ...
        1999 - ...

    :param df: dataset
    :param courses: array of courses names which will be visualised
    :return: None
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.cm.get_cmap('gist_rainbow')
    ax.set_xlabel('YEAR')
    ax.set_ylabel('HAND')
    ax.set_zlabel('MEAN')
    ax.set_title('CLUSTERS')

    # REPLACE FULL BIRTHDAY DATE WITH YEAR: "2000-03-30" -> 2000
    years = np.empty(df["Birthday"].shape[0], dtype=np.int)
    for i, b in enumerate(df["Birthday"]):
        years[i] = b.split('-')[0]
    df["Birthday"] = years

    # REPLACE HAND WITH 0/1 VALUE
    bin_hands = np.empty(df["Best Hand"].shape[0], dtype=np.int)
    for i, hand in enumerate(df["Best Hand"].unique()):
        bin_hands[df["Best Hand"] == hand] = i
    df["Best Hand"] = bin_hands

    for year in df["Birthday"].unique():
        for hand in df["Best Hand"].unique():
            for i, course in enumerate(courses):
                # CHOOSE INDEXES WHICH HAVE CLUSTERS BIRTHDAY AND BEST HAND
                mask = (df["Birthday"] == year) & (df["Best Hand"] == hand)
                cluster = np.array(df.loc[mask, course].dropna())
                # MIN - MAX SCALING
                cluster = (cluster - cluster.min()) /\
                          (cluster.max() - cluster.min())
                # CALCULATE MEAN IN CLUSTER
                mean = cluster.mean()

                if len(course) > 15:
                    course = abbreviation(course)
                ax.scatter(year, hand, mean,
                           color=cmap(round(i / courses.shape[0], 2)),
                           label=course)
    courses = [c if len(c) < 15 else abbreviation(c) for c in courses]
    ax.legend(courses)
    plt.show()


def vis_clusters_3d(data_path: str, config_path: str):
    # CHOOSE FROM CONFIG FEATURES TO PLOT
    config = Config(config_path)
    courses = config.choosed_features()

    df = pd.read_csv(data_path)

    clusters_3d(df, courses)


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--data_path', type=str,
                        default="../data/dataset_train.csv",
                        help='Path to "dataset_*.csv" file')

    parser.add_argument('--config_path', type=str,
                        default="../config.yaml",
                        help='path to .yaml file')

    args = parser.parse_args()

    vis_clusters_3d(args.data_path, args.config_path)
