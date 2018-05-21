import random
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
        if(x != col):
            distance += np.square(data1[x] - data2[x])
    return np.sqrt(distance)
def manhattanDistance(data1, data2, length):
    distance = 0
    for x in range(length):
        if(x != col):
            distance += np.absolute(data1[x] - data2[x])
    return distance

def calculateAccuracy(testData, predictedData):
    correctClasses = 0

    totalPredictionsNumber = 0
    for data in predictedData:
        if(predictedData[data] == testData.values[totalPredictionsNumber][col]):
            correctClasses += 1
        totalPredictionsNumber += 1
    
    return correctClasses*100/totalPredictionsNumber

def SetCalculateDistanceMethod(method):
    if(method == "m"):
        calculateDistanceMethod = manhattanDistance

def printResult(classes, neighbors, accuracy):
    print('')
    print("Accuracy :", str(accuracy) + "%")
    print('#########')
    for c in classes:
        print("Class :", classes[c])
        print("Neighbors :", neighbors[c])
        print("---------------------------")
    print('#########')
# -----------------------------------------------------

arg_calculateDistanceMethod = "e"
arg_test_size = 0.1
arg_col = 5
arg_k = 3

calculateDistanceMethod = euclideanDistance
col = arg_col - 1
data = pd.read_csv("KNN/iris.csv")
data = data.values
random.shuffle(data)
data = pd.DataFrame(data)

trainingData = data[:-int(arg_test_size * len(data))]
testData = data[-int(arg_test_size * len(data)):]

result, neighbors, accuracy = knn(trainingData, testData, arg_k)
printResult(result, neighbors, accuracy)