import argparse

import pandas as pd

parser = argparse.ArgumentParser(description='KNN classifier')
parser.add_argument('-k', '--knn', type=int, default=5, help='Number of k for KNN', metavar='')
parser.add_argument('-m', '--metric', default='e', help='Type of metric', metavar='')
parser.add_argument('-t', '--train', default='train', help='Type of test set', metavar='')
parser.add_argument('-f', '--file', required=True, default='iris.csv', help='Data file path', metavar='')

# parametr do programu - która kolumna jest klasową 

args = parser.parse_args()


def calculate_distance(file_name, knn, metric):
    if metric == 'm':
        print('Manhattan metric')
        data_set = pd.read_csv(file_name)
        print(data_set.head(knn))
    elif metric == 'e':
        print('Euclides metric')
        data_set = pd.read_csv(file_name)
        print(data_set.head(knn))
    else:
        print('Metric \"%s\" not found' % metric)


def KNN_classifier():
    calculate_distance(args.file, args.knn, args.metric)


if __name__ == '__main__':
    KNN_classifier()
