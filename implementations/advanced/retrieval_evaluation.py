import cPickle

from sklearn import svm
from implementations.advanced.encoder import Encoder
import numpy as np
from itertools import product
import ROOT_SCRIPT
import copy
from sklearn.preprocessing import PolynomialFeatures

ROOT_PATH = ROOT_SCRIPT.get_root_path()

def getRowIndexes(qIdListOriginal, qIdList, questionSentenceDict):
    index = 0
    start = qIdListOriginal.index(qIdList[0])
    end = qIdListOriginal.index(qIdList[-1])+1

    for qId in qIdListOriginal[:start]:
        index += len(questionSentenceDict[qId])

    startIndex = int(index)

    for qId in qIdListOriginal[start:end]:
        index += len(questionSentenceDict[qId])

    endIndex = int(index)

    return startIndex, endIndex

def transform(questionIdList, questionSentenceDict, encoder):
    transformedList = list()
    labelList = list()
    questionIds = list()
    for questionID in questionIdList:
        tripletList = questionSentenceDict[questionID]

        for index, _, label in tripletList:
            #encoded = encoder.encode(questionID, index)
            encoded = np.zeros((10,))
            transformedList.append(encoded)
            labelList.append(label)
            questionIds.append(questionID)

    return transformedList, labelList, questionIds


def makeSplits(questionSentenceDict, encoder, n_splits, randomState):
    qIdList = list(questionSentenceDict.keys())

    np.random.seed(randomState)
    np.random.shuffle(qIdList)
    print qIdList

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
        start +=testLength

    return splitList

def generateSplits(qIdList, n_splits):
    length = len(qIdList)

    onePart = length / float(n_splits)
    onePartInt = int(round(onePart))

    trainLength = (n_splits - 1) * onePartInt
    testLength = length - trainLength

    splitList = list()
    start = 0
    for i in range(n_splits):
        testIndexStart = start
        testIndexEnd = start + testLength

        start += testLength

        qId_test = qIdList[testIndexStart:testIndexEnd]
        qId_train = qIdList[0:testIndexStart] + qIdList[testIndexEnd:]
        splitList.append((qId_train, qId_test))

    return splitList


def customCompFunc(a, b):
    qId1, score1, _ = a
    qId2, score2, _ = b
    qCMP = cmp(qId1, qId2)

    if qCMP==0:
        return -cmp(score1, score2)

    return qCMP

def evaluationLoop(qIdList, qsDict, paramCombinations, shuffle=True, outer_splits=5, inner_splits=3):

    X, y, questionIdsMatchingXRows = np.load(ROOT_PATH + "data/X_data.npy"), np.load(ROOT_PATH + "data/y_targets.npy"), cPickle.load(open(ROOT_PATH+"data/q_id_list.pickle", "rb"))

    mrr_list = []
    best_params_list = []

    for qIdTrain, qIdTest in generateSplits(qIdList, outer_splits):    # outer loop

        maxResult = -1.0
        bestParams = None

        for params in paramCombinations:
            clf = svm.LinearSVC()
            clf.set_params(**params)

            current_param_scores = list()
            for qIdInnerTrain, qIdValidate in generateSplits(qIdTrain, inner_splits):
                Xtrain, yTrain, Xtest, yTest, qIdsForXRows = getInputRows(X, y, qIdList, qIdInnerTrain, qIdValidate, qsDict, questionIdsMatchingXRows)

                clf.fit(Xtrain, yTrain) #train
                yPredict = clf.decision_function(Xtest) #scores on test set

                mrr = calculateMRR(yPredict, qIdsForXRows, yTest) #calculate mrr
                current_param_scores.append(mrr)

            meanScore = np.mean(current_param_scores)
            print "+++++++++++++++++++++++++++++++++"
            print "Current params %s" % str(params)
            print "Mean MRR %f" % meanScore
            print "++++++++++++++++++++++++++++++++"
            if maxResult < meanScore:
                maxResult = meanScore
                bestParams = params

            # average results
            # if max, remember


        #trainIndexStart, trainIndexEnd = getRowIndexes(qIdList, qIdTrain, qsDict)
        #testIndexStart, testIndexEnd = getRowIndexes(qIdList, qIdTest, qsDict)
        #Xtrain = X[trainIndexStart:trainIndexEnd]
        #yTrain = y[trainIndexStart:trainIndexEnd]
        #Xtest = X[testIndexStart:testIndexEnd]
        #yTest = y[testIndexStart:testIndexEnd]

        Xtrain, yTrain, Xtest, yTest, qIdsForXRows = getInputRows(X, y, qIdList, qIdTrain, qIdTest, qsDict, questionIdsMatchingXRows)

        clf = svm.LinearSVC()
        clf.set_params(**bestParams)
        clf.fit(Xtrain, yTrain)  # train

        yPredict = clf.decision_function(Xtest)  # scores on test set

#        qIdsForXRows = questionIdsMatchingXRows[testIndexStart:testIndexEnd]  # question ids for each row in test set

        mrr = calculateMRR(yPredict, qIdsForXRows, yTest)  # calculate mrr

        print "========================================"
        print "For best params: %s " % str(bestParams)
        print "MRR score: %f" % mrr
        print "========================================"
        # take found max params
        # train on qIdTrain with max params
        # test on qIdTest
        # append result to list
        # append max params to another list
        mrr_list.append(mrr)
        best_params_list.append(bestParams)

    ## return list of results and found params list
    return mrr_list, best_params_list


def getInputRows(X, y, qIdList, qIdTrain, qIdTest, qsDict, questionIdsMatchingXRows):
    testIndexStart, testIndexEnd = getRowIndexes(qIdList, qIdTest, qsDict)

    if testIndexStart >= testIndexEnd:
        print "Fucking mistake"
        print testIndexStart, testIndexEnd

    firstX = X[0:testIndexStart]
    secondX = X[testIndexEnd:]
    firstY = y[0:testIndexStart]
    secondY = y[testIndexEnd:]

    Xtrain = np.vstack((firstX, secondX))
    yTrain = np.hstack((firstY, secondY))

    Xtest = X[testIndexStart:testIndexEnd]
    yTest = y[testIndexStart:testIndexEnd]

    # return np.concatenate([np.array([similarity, jaccard_similarity, overlap, bigram_overlap]), sentence_length, question_length, question_type, sentence_type])
    poly = PolynomialFeatures(1)
    qIdsForXRows = questionIdsMatchingXRows[testIndexStart:testIndexEnd]  # question ids for each row in test set

    Xtrain = poly.fit_transform(Xtrain)
    Xtest = poly.fit_transform(Xtest)


    return Xtrain, yTrain, Xtest, yTest, qIdsForXRows


def calculateMRR(yPredict, qIdsForXRows, yTestLabels):
    qScoreDict = {}
    for i in range(len(yPredict)):
        qId, score, label = qIdsForXRows[i], yPredict[i], yTestLabels[i]

        if qId not in qScoreDict:
            qScoreDict[qId] = [(score, label)]

        else:
            qScoreDict[qId].append((score, label))

    mrrSum = 0.0

    for qId in qScoreDict.keys():
        sortedList = sorted(qScoreDict[qId], key=lambda x: x[0], reverse=True)

        for i in range(min(len(sortedList), 20)):
            score, label = sortedList[i]

            if label == "1" or label == 1:
                mrrSum += (1 / float(i+1))
                break

    return mrrSum / float(len(qScoreDict.keys()))

def create_data(seed=27):
    with open(ROOT_PATH + "pickles/question_labeled_sentence_dict.pickle", "rb") as f:
        qsDict = cPickle.load(f)
    keys = qsDict.keys()
    np.random.seed(seed)
    np.random.shuffle(keys)
    encoder = Encoder()
    encoder.create_structures()
    X_data, y_targets, q_id_list = [], [], []
    shuffled_IDs = []
    for key in keys:
        shuffled_IDs.append(key)
        for index, text, label in qsDict[key]:
            feature_vector = encoder.encode(str(key), int(index))
            X_data.append(feature_vector)
            y_targets.append(int(label))
            q_id_list.append(key)

    X_data, y_targets = np.array(X_data), np.array(y_targets)
    np.save(open(ROOT_PATH + "data/X_data.npy", "wb"), X_data)
    np.save(open(ROOT_PATH + "data/y_targets.npy", "wb"), y_targets)
    cPickle.dump(shuffled_IDs, open(ROOT_PATH+"data/shuffled_IDs.pickle", "wb"))
    cPickle.dump(q_id_list, open(ROOT_PATH+"data/q_id_list.pickle", "wb"))

def baselineEvaluationLoop(qIdList, qsDict, paramCombinations, shuffle=True, outer_splits=5, inner_splits=3):

    X, y, questionIdsMatchingXRows = np.load(ROOT_PATH + "data/X_data.npy"), np.load(ROOT_PATH + "data/y_targets.npy"), cPickle.load(open(ROOT_PATH+"data/q_id_list.pickle", "rb"))

    mrr_list = []

    for qIdTrain, qIdTest in generateSplits(qIdList, outer_splits):    # outer loop

        Xtrain, yTrain, Xtest, yTest, qIdsForXRows = getInputRows(X, y, qIdList, qIdTrain, qIdTest, qsDict, questionIdsMatchingXRows)

        yPredict = Xtest[:, 0] # scores on test set
        print Xtrain.shape, Xtest.shape

#        qIdsForXRows = questionIdsMatchingXRows[testIndexStart:testIndexEnd]  # question ids for each row in test set

        mrr = calculateMRR(yPredict, qIdsForXRows, yTest)  # calculate mrr

        mrr_list.append(mrr)

    return mrr_list

def temporaryLoop(qIdList, qsDict, paramCombinations, shuffle=True, outer_splits=5, inner_splits=3):

    X, y, questionIdsMatchingXRows = np.load(ROOT_PATH + "data/X_data.npy"), np.load(ROOT_PATH + "data/y_targets.npy"), cPickle.load(open(ROOT_PATH+"data/q_id_list.pickle", "rb"))

    mrr_list = []

    for qIdTrain, qIdTest in generateSplits(qIdList, outer_splits):    # outer loop

        Xtrain, yTrain, Xtest, yTest, qIdsForXRows = getInputRows(X, y, qIdList, qIdTrain, qIdTest, qsDict, questionIdsMatchingXRows)
        clf = svm.LinearSVC(C=2**-8, class_weight={1: 10})
        clf.fit(Xtrain, yTrain)  # train
        yPredict = clf.decision_function(Xtest)

#        qIdsForXRows = questionIdsMatchingXRows[testIndexStart:testIndexEnd]  # question ids for each row in test set

        mrr = calculateMRR(yPredict, qIdsForXRows, yTest)  # calculate mrr

        mrr_list.append(mrr)

    return mrr_list


#enc = Encoder()
#enc = None
#makeSplits(qsDict, enc, 5, 42)

with open(ROOT_PATH +"pickles/question_labeled_sentence_dict.pickle", "rb") as f:
    qsDict = cPickle.load(f)
qIdList = cPickle.load(open(ROOT_PATH+"data/shuffled_IDs.pickle", "rb"))

params = []
clist = range(-9, -4)
classweights = [6, 8]
#
for c, w in product(clist, classweights):
    params.append({"C":2**c, "class_weight":{1:w}})

mrr_scores, best_params_list = evaluationLoop(qIdList, qsDict, params, inner_splits=3, outer_splits=5)
#
print "Final Results: "
print mrr_scores
print best_params_list

#create_data(58)
#
#result = temporaryLoop(qIdList, qsDict, [])
#
#print result
#print np.mean(result)
