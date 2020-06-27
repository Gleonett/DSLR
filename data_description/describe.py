"""
Display information for all numerical features
"""

from argparse import ArgumentParser

from data_describer import HogwartsDataDescriber


def abbreviation(string: str) -> str:
    """
    Abbreviation for title
    :param string:
    :return: str
    """
    string_list = string.split(" ")
    abb = ""
    for word in string_list:
        abb += word[0]
    return abb


def describe(csv_path: str):
    data = HogwartsDataDescriber.read_csv(csv_path)
    print(f'{"":15} |{"Count":>12} |{"Mean":>12} |{"Std":>12} |{"Min":>13}'
          f'|{"Max":>12} |{"25%":>12} |{"50%":>12} |{"75%":>12} |')
    for feature in data.columns:
        if len(feature) > 15:
            print(f'{abbreviation(feature):15.15}', end=' |')
        else:
            print(f'{feature:15.15}', end=' |')
        print(f'{data.count(feature):>12.4f}', end=' |')
        if data.is_numeric(feature) and data.count(feature) != 0:
            # print(f'{data.count(feature):>12.4f}', end=' |')
            print(f'{data.mean(feature):>12.4f}', end=' |')
            print(f'{data.std(feature):>12.4f}', end=' |')
            print(f'{data.min(feature):>12.4f}', end=' |')
            print(f'{data.max(feature):>12.4f}', end=' |')
            print(f'{data.percentile(feature, 25):>12.4f}', end=' |')
            print(f'{data.percentile(feature, 50):>12.4f}', end=' |')
            print(f'{data.percentile(feature, 75):>12.4f}', end=' |\n')
        else:
            print(f'{"No numerical value to display":>64}')


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('data_path', type=str, help='Path to .csv file')

    args = parser.parse_args()

    describe(args.data_path)
