import operator

import numpy as np
import pandas as pd

# -----------------------------------------------------
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
        number+=1


    return predictedClassesForTests, neighborsForTests

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

def printResult(classes, neighbors):
    print('')
    print('#########')
    print('Classes:')
    print(classes)
    print('Neighbors: ')
    print(neighbors)
    print('#########')
# -----------------------------------------------------

k = 3
data = pd.read_csv("KNN/iris.csv")

testSet = [[7.2, 3.6, 5.1, 2.5],
           [1.2, 3.6, 5.1, 5.5],
           [7.2, 2.6, 5.1, 2.5]]

test = pd.DataFrame(testSet)

calculateDistanceMethod = euclideanDistance
# calculateDistanceMethod = manhattanDistance

result, neighbors = knn(data, test, k)

printResult(result, neighbors)