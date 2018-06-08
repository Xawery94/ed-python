import argparse
import math

import pandas as pd
from graphviz import Digraph

parser = argparse.ArgumentParser(description='Decision Tree classifier')
parser.add_argument('-l', '--decisionLabel', type=int, default=4, help='Decision column', metavar='')
parser.add_argument('-ih', '--isHeader', action='store_true', help='Is columns in file has headers')
parser.add_argument('-a', '--decisionAmount', type=int, default=2, help='Amount of decision classes', metavar='')
parser.add_argument('-i', '--minInfoGain', type=float, default=0.0, help='Min InfoGain', metavar='')
parser.add_argument('file', type=argparse.FileType('r'), metavar='FILE')

args = parser.parse_args()


def id3(data, decisionEntropy):

    print("Calculate entropy")
    entropies = calculateEntrophy(data)

    if (decisionEntropy == None):
        decisionEntropy = entropies[decisionLabel]

    print("Calculate InfoGains")
    infoGains = calculateInfoGain(entropies, decisionEntropy)
    if(len(infoGains) != 0):
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

            if(infoGain > minInfoGain):
                result[e] = {"entropy": entropies[e]["entropy"],
                            "decisions": entropies[e]["decisions"],
                            "infoGain": infoGain}

    return result


def getRoot(data, infoGains, decisionEntropy):
    rootCandidate = {}

    for ig in infoGains:
        if ("infoGain" not in rootCandidate or infoGains[ig]["infoGain"] > rootCandidate["infoGain"]):
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

def getMostFrequentDecision(dictionary):
    candidateName = ''
    candidateCount = 0
    for d in dictionary:
        if(candidateName == '' or dictionary[d] > candidateCount):
            candidateName = d
            candidateCount = dictionary[d]
    return candidateName


def getNewData(data, label, rowLabel):
    newData = data[data[label] == rowLabel]
    return newData.drop([label], axis=1)

def checkIfAttributeExistsInDictionary(attribute, dictionary):
    for d in dictionary:
        if(d == attribute):
            return True
    
    return False

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

    if(len(root["leafs"]) > 0):
        rows = GetRows(data, root["label"])
        i = 0
        for r in rows:
            if(checkIfAttributeExistsInDictionary(r[0], root["nodes"]) == False):
                label = ''
                if(checkIfAttributeExistsInDictionary(r[0], root["leafs"]) == True):
                    label = root["leafs"][r[0]]
                else:
                    label = getMostFrequentDecision(decisionsCount)

                leafName = str(i) + str(rootName) + str(label)
                g.node(str(leafName), str(label))
                g.edge(str(rootName), str(leafName), label=str(r[0]))
            i += 1

    if(len(root["nodes"]) > 0):
        i = 0
        for n in root["nodes"]:
            if(root["nodes"][n] != None):
                nodeLabel = str(root["nodes"][n]["label"])
                nodeName = str(i) + str(rootName) + str(nodeLabel)
                g.node(str(nodeName), str(nodeLabel))
                g.edge(str(rootName), str(nodeName), label=str(n))

                decisions = GetDecisionRows(data, root["label"], n)
                for d in decisions:
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
data = None
if(args.isHeader == True):
    data = pd.read_csv(args.file)
else:
    data = pd.read_csv(args.file, header=None)

arg_nDecisionClasses = args.decisionAmount
decisionLabel = data.columns[args.decisionLabel - 1]
minInfoGain = args.minInfoGain
# -------------------------

if __name__ == '__main__':
    print("Run ID3")
    root = id3(data, None)

    if(root != None):
        printTree(root, None, None, None)
    else:
        print("Cant generate tree. Cant find root. Change minimum info gain.")
