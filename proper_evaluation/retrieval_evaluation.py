import cPickle

from implementations.advanced.encoder import Encoder
import numpy as np
import ROOT_SCRIPT

ROOT_PATH = ROOT_SCRIPT.get_root_path()


def getRowIndexes(qIdList, start, end, questionSentenceDict):
    index = 0
    for qId in qIdList[:start]:
        index += len(questionSentenceDict[qId])

    startIndex = index

    for qId in qIdList[start:end]:
        index += len(questionSentenceDict[qId])

    endIndex = index

    return startIndex, endIndex

def transform(questionIdList, questionSentenceDict, encoder):
    transformedList = list()
    labelList = list()

    for questionID in questionIdList:
        tripletList = questionSentenceDict[questionID]

        for index, _, label in tripletList:
            encoded = encoder.encode(questionID, index)
            transformedList.append(encoded)
            labelList.append(label)

    return transformedList, labelList


def makeSplits(questionSentenceDict, encoder, n_splits, randomState):
    #qIdList = list(questionSentenceDict.keys())
    qIdList = ["201", "202", "203", "204", "205"]

    np.random.seed(randomState)
    np.random.shuffle(qIdList)

    length = len(qIdList)

    onePart = length / float(n_splits)
    onePartInt = int(round(onePart))

    X, y = transform(qIdList, questionSentenceDict, encoder)

    trainLength = (n_splits - 1) * onePartInt
    testLength = length - trainLength

    start = 0
    splitList = list()
    for i in range(n_splits):
        rowStartTest, rowEndTest = getRowIndexes(qIdList, start, start + testLength, questionSentenceDict)

        X_test = X[rowStartTest:rowEndTest]
        X_train = X[0:rowStartTest] + X[rowEndTest:]
        y_test = y[rowStartTest:rowEndTest]
        y_train = y[0:rowStartTest] + y[rowEndTest:]

        print "TRAIN LEN: %d" % len(X_train)
        print "TEST LEN: %d" % len(X_test)

        splitList.append((X_train, y_train, X_test, y_test))

    return splitList

with open(ROOT_PATH + "pickles/question_labeled_sentence_dict.pickle", "rb") as f:
    qsDict = cPickle.load(f)

enc = Encoder()
makeSplits(qsDict, enc, 5, 42)
