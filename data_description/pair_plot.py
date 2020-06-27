"""
Show pair plots for all courses
"""

import matplotlib.pyplot as plt
from argparse import ArgumentParser

from histogram import histogram
from scatter_plot import scatter_plot
from data_describer import HogwartsDataDescriber


def show_pair_plot(csv_path: str):
    # obtaining data for plotting
    df = HogwartsDataDescriber.read_csv(csv_path)
    courses = list(df.columns[6:])

    _, axs = plt.subplots(13, 13, figsize=(25.6, 14.4), tight_layout=True)
    for row_course, row_plt in zip(courses, axs):
        for col_course, col_plt in zip(courses, row_plt):
            # plotting
            if row_course == col_course:
                histogram(col_plt, df, row_course)
            else:
                scatter_plot(col_plt, df, row_course, col_course)

            # remove values from axis
            col_plt.tick_params(labelbottom=False)
            col_plt.tick_params(labelleft=False)

            # set x labels
            if col_plt.is_last_row():
                col_plt.set_xlabel(col_course.replace(' ', '\n'))

            # set y labels
            if col_plt.is_first_col():
                label = row_course.replace(' ', '\n')
                length = len(label)
                if length > 14 and '\n' not in label:
                    label = label[:int(length/2)] + "\n" + \
                            label[int(length/2):]
                col_plt.set_ylabel(label)

    plt.legend(df.houses,
               loc='center left',
               frameon=False,
               bbox_to_anchor=(1, 0.5))
    plt.show()


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--data_path',
                        type=str,
                        default='../data/dataset_train.csv',
                        help='Path to dataset_train.csv file')

    args = parser.parse_args()

    show_pair_plot(args.data_path)
