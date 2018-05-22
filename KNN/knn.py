import argparse
import operator
import random

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='KNN classifier')
parser.add_argument('-k', '--knn', type=int, default=5, help='Number of k for KNN', metavar='')
parser.add_argument('-m', '--metric', default='e', help='Type of metric [e|m]', metavar='')
parser.add_argument('-t', '--train', default='train', help='Type of test set [train|split]', metavar='')
parser.add_argument('-d', '--decision', required=True, type=int, default=1, help='Index of decision attribute',
                    metavar='')
parser.add_argument('-s', '--split', default=0.5, type=float, help='Split size in percent', metavar='')
parser.add_argument('file', type=argparse.FileType('r'), metavar='FILE')

group = parser.add_mutually_exclusive_group()
group.add_argument('-v', '--verbose', action='store_true', help='Print all data')

args = parser.parse_args()


def knn(trainingSet, testInstance, k):
    predictedClassesForTests = {}
    neighborsForTests = {}

    number = 0
    for testRow in testInstance.values:
        distances = calculateDistancesToAllTrainingSets(trainingSet, testRow)
        neighbors = getClosestNeighbors(distances, k)
        mostFrequentClass = getMostFrequentClass(trainingSet, neighbors)

        neighborsForTests[number] = neighbors
        predictedClassesForTests[number] = mostFrequentClass

        number += 1

    accuracy = calculateAccuracy(testInstance, predictedClassesForTests)
    return predictedClassesForTests, neighborsForTests, accuracy


def calculateDistancesToAllTrainingSets(trainingSet, testInstance):
    objectArgumentsCount = testInstance.shape[0]
    distances = {}

    for x in range(len(trainingSet)):
        dist = calculateDistanceMethod(testInstance, trainingSet.iloc[x], objectArgumentsCount)
        distances[x] = dist

    return distances


def getClosestNeighbors(distanceList, k):
    sortedDistance = sorted(distanceList.items(), key=operator.itemgetter(1))
    neighbors = []

    for x in range(k):
        neighbors.append(sortedDistance[x][0])

    return neighbors


def getMostFrequentClass(trainingSet, neighbors):
    classVotes = {}

    for x in range(len(neighbors)):
        response = trainingSet.iloc[neighbors[x]][col]

        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1

    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)

    return sortedVotes[0][0]


def euclideanDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        if x != col:
            distance += np.square(data1[x] - data2[x])
    return np.sqrt(distance)


def manhattanDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        if x != col:
            distance += np.absolute(data1[x] - data2[x])
    return distance


def calculateAccuracy(testData, predictedData):
    correctClasses = 0

    totalPredictionsNumber = 0
    for data in predictedData:
        if predictedData[data] == testData.values[totalPredictionsNumber][col]:
            correctClasses += 1
        totalPredictionsNumber += 1

    return correctClasses * 100 / totalPredictionsNumber


def SetCalculateDistanceMethod(method):
    if method == "m":
        return manhattanDistance
    elif method == "e":
        return euclideanDistance


def printResult(classes, neighbors, accuracy):
    print('')
    print('#########')
    for c in classes:
        print("Class :", classes[c])
        if args.verbose:
            print("Neighbors :", neighbors[c])
            print("---------------------------")
    print('#########')
    print("Accuracy :", str(accuracy) + "%")


# -----------------------------------------------------

arg_calculateDistanceMethod = args.metric
arg_test_size = args.split
arg_col = args.decision
arg_k = args.knn
arg_file = args.file
arg_train = args.train

# -----------------------------------------------------

calculateDistanceMethod = SetCalculateDistanceMethod(arg_calculateDistanceMethod)
col = arg_col - 1
data = pd.read_csv(arg_file)
data = data.values
random.shuffle(data)
data = pd.DataFrame(data)

if arg_train == 'split':
    trainingData = data[:-int(arg_test_size * len(data))]
    testData = data[-int(arg_test_size * len(data)):]
elif arg_train == 'train':
    trainingData = data
    testData = data

if __name__ == '__main__':
    result, neighbors, accuracy = knn(trainingData, testData, arg_k)
    printResult(result, neighbors, accuracy)
