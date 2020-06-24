"""
Plot course marks distribution
"""

import matplotlib.pyplot as plt
from argparse import ArgumentParser

from data_describer import HogwartsDataDescriber


def histogram(plot: plt,
              df: HogwartsDataDescriber,
              course: str):
    """
    Scatter plot for 2 courses
    :param plot: matplotlib.axes._subplots.AxesSubplot
    :param df: HogwartsDataDescriber
    :param course: course name
    :return: None
    """
    for house, color in zip(df.houses, df.colors):
        # choose course marks of students belonging to the house
        marks = df[course][df['Hogwarts House'] == house].dropna()

        plot.hist(marks, color=color, alpha=0.5)


def show_course_marks_distribution(csv_path: str, course: str):
    # obtaining data for plotting
    df = HogwartsDataDescriber.read_csv(csv_path)
    _, ax = plt.subplots()

    histogram(ax, df, course)
    ax.set_title(course)
    ax.legend(df.houses, frameon=False)
    ax.set_xlabel('Marks')
    ax.set_ylabel('Students')
    plt.show()


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--data_path',
                        type=str,
                        default='../data/dataset_train.csv',
                        help='Path to dataset_train.csv file')

    parser.add_argument('--course',
                        type=str,
                        default='Care of Magical Creatures',
                        help='Name of the course to plot')

    args = parser.parse_args()

    show_course_marks_distribution(args.data_path, args.course)
