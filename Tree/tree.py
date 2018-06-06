import argparse
import operator
import random
import math
from collections import Counter
import itertools

import numpy as np
import pandas as pd

def id3(data, decisionEntropy):
    entropies = calculateEntrophy(data)

    if(decisionEntropy == None):
        decisionEntropy = entropies[decisionLabel]

    infoGains = calculateInfoGain(entropies, decisionEntropy)
    root = getRoot(data, infoGains, decisionEntropy)

    return root

def calculateEntrophy(data):
    dataLength = len(data)
    entropies = {}

    for col in data.columns:
        rows = GetRows(data, col)
        entropy = 0
        decisions = {}
        for row in rows:
            decisionValues = GetDecisionRows(data, col, row[0])
            rowValue = row[1] / dataLength
            result = 0
            if(decisionLabel != col):
                entropy += rowValue * GetResult(decisionValues, row[1])
            else:
                entropy += GetResult(decisionValues, dataLength)

            decisions[row[0]] = decisionValues

        entropies[col] = {"entropy": entropy, "decisions": decisions}
    return entropies
        
def calculateInfoGain(entropies, decisionEntropy):
    decisionValue = decisionEntropy
    result = {}

    for e in entropies:
        if(e != decisionLabel):
            infoGain = decisionValue["entropy"] - entropies[e]["entropy"]
            result[e] = {"entropy": entropies[e]["entropy"], "decisions": entropies[e]["decisions"], "infoGain": infoGain}

    return result

def getRoot(data, infoGains, decisionEntropy):
    rootCandidate = {"infoGain": 0}

    for ig in infoGains:
        if(infoGains[ig]["infoGain"] > rootCandidate["infoGain"]):
            rootCandidate = {"entropy": infoGains[ig]["entropy"], "decisions": infoGains[ig]["decisions"], "infoGain": infoGains[ig]["infoGain"], "label": ig, "leafs":{}, "nodes":{}}
    
    for d in rootCandidate["decisions"]:
        if(len(rootCandidate["decisions"][d]) == 1):
            rootCandidate["leafs"][d] = rootCandidate["decisions"][d][0][0]
        else:
            print("NODE")
            newData = getNewData(data, rootCandidate["label"], d)
            rootCandidate["nodes"][d] = id3(newData, decisionEntropy)
            
    return rootCandidate

def GetResult(decisionValues, count):
    result = 0
    for label in decisionValues:
        y = label[1] / count
        if y != 0:
            result -= y * math.log(y, 2)
    return result

def GetGroupedList(data, column):
    return data.groupby([column, decisionLabel]).size().reset_index(name='Count').values

def GetRows(data, column):
    return data.groupby(column).size().reset_index(name='Count').values

def GetDecisionRows(data, rowColumn, row):
    return data[data[rowColumn] == row].groupby(decisionLabel).size().reset_index(name='Count').values

def getNewData(data, label, rowLabel):
    newData = data[data[label] == rowLabel]
    return newData.drop([label], axis=1)

def printTree(root):
    print("Print root here!!!!!!!!!!!")

arg_file = "s.csv"
arg_isHeader = True
arg_decisionIndex = 3

data = pd.read_csv(arg_file)
decisionLabel = data.columns[arg_decisionIndex]
root = id3(data, None)
printTree(root)

print("test")