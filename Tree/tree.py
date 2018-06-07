import argparse
import math

import pandas as pd
from graphviz import Digraph

parser = argparse.ArgumentParser(description='Decision Tree classifier')
parser.add_argument('-l', '--decisionLabel', type=int, default=3, help='Decision column', metavar='')
parser.add_argument('-a', '--decisionAmount', type=int, default=4, help='Amount of decision classes', metavar='')
# parser.add_argument('file', type=argparse.FileType('r'), metavar='FILE')  #uncomment at the end

args = parser.parse_args()


def id3(data, decisionEntropy):

    print("Calculate entropy")
    entropies = calculateEntrophy(data)

    if (decisionEntropy == None):
        decisionEntropy = entropies[decisionLabel]

    print("Calculate InfoGains")
    infoGains = calculateInfoGain(entropies, decisionEntropy)
    root = getRoot(data, infoGains, decisionEntropy)

    print("Return root")
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
    rootCandidate = None

    for ig in infoGains:
        if (rootCandidate == None or infoGains[ig]["infoGain"] > rootCandidate["infoGain"]):
            rootCandidate = {"entropy": infoGains[ig]["entropy"],
                             "decisions": infoGains[ig]["decisions"],
                             "infoGain": infoGains[ig]["infoGain"],
                             "label": ig,
                             "leafs": {},
                             "nodes": {}}

    for d in rootCandidate["decisions"]:
        if (len(rootCandidate["decisions"][d]) == 1):
            rootCandidate["leafs"][d] = rootCandidate["decisions"][d][0][0]
        else:
            newData = getNewData(data, rootCandidate["label"], d)
            rootCandidate["nodes"][d] = id3(newData, decisionEntropy)

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


def printTree(root, g, rootName, decisionsCount):
    isRoot = False
    if(rootName == None):
        rootName = root["label"]
        isRoot = True
    if(g == None):
        g = Digraph()
        g.node(str(rootName), str(rootName))
    if(decisionsCount == None):
        decisionsCount = {}

    if(len(root["leafs"]) == 0):
        i = 0
        for l in root["leafs"]:
            label = root["leafs"][l]
            leafName = str(i) + str(rootName) + str(label)
            g.node(str(leafName), str(label))
            g.edge(str(rootName), str(leafName), label=str(l))
            i += 1

    # if(parentMostFrequentDecision != None):

    if(len(root["nodes"]) > 0):
        i = 0
        for n in root["nodes"]:
            nodeLabel = root["nodes"][n]["label"]
            nodeName = str(i) + str(rootName) + str(nodeLabel)
            g.node(str(nodeName), str(nodeLabel))
            g.edge(str(rootName), str(nodeName), label=str(n))

            for d in root["decisions"][n]:
                if(d[0] in decisionsCount):
                    decisionsCount[d[0]] += d[1]
                else:
                    decisionsCount[d[0]] = d[1]

            printTree(root["nodes"][n], g, nodeName, decisionsCount)
            i += 1

    if(isRoot == True):
        print("Print tree.")
        g.format = "png"
        g.render('output/tree.gv', view=True)


# -------------------------
arg_isHeader = None
arg_nDecisionClasses = args.decisionAmount
# data = pd.read_csv(args.file)    #file import
data = pd.read_csv('car.csv', header=arg_isHeader)
# decisionLabel = data.columns[args.decisionLabel]
decisionLabel = 6
# -------------------------

if __name__ == '__main__':
    print("Run ID3")
    root = id3(data, None)

    printTree(root, None, None, None)
