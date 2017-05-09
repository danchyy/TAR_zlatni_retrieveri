import cPickle
from time import time

import numpy as np
import re

from implementations.baseline.answer_extraction import AnswerExtraction
from implementations.baseline.answer_retrieval import AnswerRetrieval
from implementations.baseline.preprocessing import Preprocessing

def calculateRR(questionIndex, retrievedSentences):
    rank = 1
    for sentence in retrievedSentences:
        if sentence.question_ID == questionIndex and sentence.label == "1":
            return 1.0 / float(rank)

        rank += 1

    return 0.0

def matches(answer, pattern):
    prog = re.compile(pattern)
    return prog.match(answer) is not None

def calculateMatch(answer, patterns):
    for pattern in patterns:
        if matches(answer, pattern):
            return 1.0

    return 0.0

t0 = time()
with open("/home/marin/tar_pickles/temp_pickle.p", "rb") as f:
    uberDict = cPickle.load(f)
t1 = time()

print "elapsed: %.3f" % (t1 - t0)

with open("/home/marin/tar_pickles/questions.pickle", "rb") as f:
    questionDict = cPickle.load(f)

with open("/home/marin/tar_pickles/patterns.pickle", "rb") as f:
    patternDict = cPickle.load(f)


questionIndexes = np.random.choice(a=range(201, 790), size=200)

sentences = uberDict.values()

uberDict = None

preprocessing = Preprocessing()
preprocessing.loadParser()
answerRetrieval = AnswerRetrieval()
answerExtraction = AnswerExtraction()

RR_list = list()
zero_one_list = list()
for questionIndex in questionIndexes:
    questionString = questionDict[questionIndex]
    question = preprocessing.rawTextToSentences(questionString)[0]

    retrievedSentences = answerRetrieval.retrieve(question, sentences)

    RR_list.append(calculateRR(questionIndex, retrievedSentences))

    answer = answerExtraction.extract(question, retrievedSentences)

    zero_one_list.append(calculateMatch(answer + "\n", patternDict[questionIndex]))

print "MRR: %.3f" % (np.average(RR_list))
print "Zero one loss: %.3f" % (np.average(zero_one_list))
