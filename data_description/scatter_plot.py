"""
Scatter plot of 2 courses
"""

import matplotlib.pyplot as plt
from argparse import ArgumentParser

from data_describer import HogwartsDataDescriber


def scatter_plot(plot: plt,
                 df: HogwartsDataDescriber,
                 course1: str,
                 course2: str):
    """
    Scatter plot for 2 courses
    :param plot: matplotlib.axes._subplots.AxesSubplot
    :param df: HogwartsDataDescriber
    :param course1: course 1 name
    :param course2: course 2 name
    :return: None
    """

    for house, color in zip(df.houses, df.colors):
        # choose course marks of students belonging to the house
        x = df[course1][df['Hogwarts House'] == house]
        y = df[course2][df['Hogwarts House'] == house]

        plot.scatter(x, y, color=color, alpha=0.5)


def show_scatter_plot(csv_path: str, course1: str, course2: str):
    # obtaining data for plotting
    df = HogwartsDataDescriber.read_csv(csv_path)
    _, ax = plt.subplots()

    scatter_plot(ax, df, course1, course2)
    ax.set_xlabel(course1)
    ax.set_ylabel(course2)
    ax.legend(df.houses)
    plt.show()


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--data_path',
                        type=str,
                        default='../data/dataset_train.csv',
                        help='Path to dataset_train.csv file')

    parser.add_argument('--course1',
                        type=str,
                        default='Astronomy',
                        help='Name of the course for x axis')

    parser.add_argument('--course2',
                        type=str,
                        default='Defense Against the Dark Arts',
                        help='Name of the course for y axis')

    args = parser.parse_args()

    show_scatter_plot(args.data_path, args.course1, args.course2)
