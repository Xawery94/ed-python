import operator

import numpy as np
import pandas as pd

def knn(trainingSet, testInstance, k):
    distances = calculateDistancesToAllTrainingSets(trainingSet, testInstance)
    neighbors = getClosestNeighbors(distances, k)
    mostFrequentClass = getMostFrequentClass(trainingSet, neighbors)

    return mostFrequentClass, neighbors

def calculateDistancesToAllTrainingSets(trainingSet, testInstance):
    length = testInstance.shape[1]
    distances = {}

    for x in range(len(trainingSet)):
        dist = calculateDistanceMethod(testInstance, trainingSet.iloc[x], length)
        # dist = manhattanDistance(testInstance, trainingSet.iloc[x], length)

        #TODO How to predict more than one row?? Maybe for?
        distances[x] = dist[0]

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
        response = trainingSet.iloc[neighbors[x]][-1]

        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)

    return sortedVotes[0][0]
  
def euclideanDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += np.square(data1[x] - data2[x])
    return np.sqrt(distance)
def manhattanDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        distance += np.absolute(data1[x] - data2[x])
    return distance
# ---------------------------------------------------------

data = pd.read_csv("KNN/iris.csv")

testSet = [[7.2, 3.6, 5.1, 2.5],
           [1.2, 3.6, 5.1, 5.5],
           [7.2, 2.6, 5.1, 2.5]]
test = pd.DataFrame(testSet)

calculateDistanceMethod = euclideanDistance
# calculateDistanceMethod = manhattanDistance

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
