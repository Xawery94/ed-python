import argparse
import math

import pandas as pd

parser = argparse.ArgumentParser(description='Decision Tree classifier')
parser.add_argument('-l', '--decisionLabel', type=int, default=3, help='Decision column', metavar='')
parser.add_argument('-a', '--decisionAmount', type=int, default=2, help='Amount of decision classes', metavar='')
# parser.add_argument('file', type=argparse.FileType('r'), metavar='FILE')  #uncomment at the end

args = parser.parse_args()


def id3(data, decisionEntropy):
    entropies = calculateEntrophy(data)

    if (decisionEntropy == None):
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
            if (decisionLabel != col):
                entropy += rowValue * GetResult(decisionValues, row[1])
            else:
                entropy += GetResult(decisionValues, dataLength)

            decisions[row[0]] = decisionValues

        entropies[col] = {"entropy": entropy,
                          "decisions": decisions}
    return entropies


def calculateInfoGain(entropies, decisionEntropy):
    decisionValue = decisionEntropy
    result = {}

    for e in entropies:
        if (e != decisionLabel):
            infoGain = decisionValue["entropy"] - entropies[e]["entropy"]
            result[e] = {"entropy": entropies[e]["entropy"],
                         "decisions": entropies[e]["decisions"],
                         "infoGain": infoGain}

    return result


def getRoot(data, infoGains, decisionEntropy):
    rootCandidate = {"infoGain": 0}

    for ig in infoGains:
        if (infoGains[ig]["infoGain"] > rootCandidate["infoGain"]):
            rootCandidate = {"entropy": infoGains[ig]["entropy"],
                             "decisions": infoGains[ig]["decisions"],
                             "infoGain": infoGains[ig]["infoGain"],
                             "label": ig,
                             "leafs": {},
                             "nodes": {}}

    for d in rootCandidate["decisions"]:
        if (len(rootCandidate["decisions"][d]) == 1):
            rootCandidate["leafs"][d] = rootCandidate["decisions"][d][0][0]
            print('L: ' + rootCandidate['label'])
        else:
            # print("NODE")
            newData = getNewData(data, rootCandidate["label"], d)
            rootCandidate["nodes"][d] = id3(newData, decisionEntropy)
            print('N: ' + rootCandidate['label'])

    return rootCandidate


def GetResult(decisionValues, count):
    result = 0
    for label in decisionValues:
        y = label[1] / count
        if y != 0:
            result -= y * math.log(y, arg_nDecisionClasses)
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
    print(root)
    print("Print root here!!!!!!!!!!!")


# -------------------------
arg_isHeader = True
arg_nDecisionClasses = args.decisionAmount
# data = pd.read_csv(args.file)    #file import
data = pd.read_csv('s.csv')
decisionLabel = data.columns[args.decisionLabel]
# -------------------------

if __name__ == '__main__':
    root = id3(data, None)
    printTree(root)
