import operator

import numpy as np
import pandas as pd

data = pd.read_csv("iris.csv")


def euclideanDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += np.square(data1[x] - data2[x])
    return np.sqrt(distance)


def knn(trainingSet, testInstance, k):
    distances = {}
    length = testInstance.shape[1]

    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet.iloc[x], length)

        #TODO How to predict more than one row?? Maybe for?
        distances[x] = dist[0]

    sorted_d = sorted(distances.items(), key=operator.itemgetter(1))

    neighbors = []

    # Extracting top k neighbors
    for x in range(k):
        neighbors.append(sorted_d[x][0])

    classVotes = {}

    # Calculating the most freq class in the neighbors
    for x in range(len(neighbors)):
        response = trainingSet.iloc[neighbors[x]][-1]

        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1

    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0], neighbors


testSet = [[7.2, 3.6, 5.1, 2.5],
           [1.2, 3.6, 5.1, 5.5],
           [7.2, 2.6, 5.1, 2.5]]
test = pd.DataFrame(testSet)

# Setting number of neighbors
k = 3
# Running KNN model
result, neigh = knn(data, test, k)

# Predicted class
print('')
print('####')
print(result)

# Nearest neighbor
print(neigh)
