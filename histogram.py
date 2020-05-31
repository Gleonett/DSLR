"""
Plot course marks distribution
"""

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from data_analysis.data import HogwartsDataDescriber


def show_course_marks_distribution(csv_path: str):
    # obtaining data for plotting
    df = HogwartsDataDescriber.read_csv(csv_path)
    houses = sorted(set(df['Hogwarts House']))
    colors = ['red', 'blue', 'green', 'yellow']
    courses = list(df.columns[6:]) + [None] * 3
    courses = np.array(courses).reshape(4, 4)

    fig, axs = plt.subplots(4, 4, figsize=(25.6, 14.4), tight_layout=True)
    for row_courses, row_plt in zip(courses, axs):
        for course, col_plt in zip(row_courses, row_plt):
            if not course:
                fig.delaxes(col_plt)
                continue

            col_plt.set_title(course)
            for house, color in zip(houses, colors):
                # choose marks of students belonging to the house
                marks = df[course][df['Hogwarts House'] == house].dropna()
                # plot marks
                col_plt.hist(marks, color=color, alpha=0.5)
                col_plt.legend(houses, frameon=False)
                col_plt.set_xlabel('Marks')
                col_plt.set_ylabel('Students')
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_path',
                        type=str,
                        default='data/dataset_train.csv',
                        help='Path to dataset_train.csv file')
    args = parser.parse_args()

    show_course_marks_distribution(args.data_path)
