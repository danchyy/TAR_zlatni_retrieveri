import cPickle
from time import time

import numpy as np
import re

from implementations.baseline.answer_retrieval import AnswerRetrieval
from implementations.baseline.preprocessing import Preprocessing
from implementations.baseline.anwer_extraction import BaselineAnswerExtraction

def calculateRR(questionIndex, retrievedSentences):
    rank = 1
    for score, sentence in retrievedSentences:
        if sentence.question_ID == str(questionIndex) and str(sentence.label).strip() == "1":
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
with open("../../pickles/sentences_serialized.p", "rb") as f:
    uberDict = cPickle.load(f)
t1 = time()

print "elapsed: %.3f" % (t1 - t0)

with open("../../pickles/questions.pickle", "rb") as f:
    questionDict = cPickle.load(f)

with open("../../pickles/patterns.pickle", "rb") as f:
    patternDict = cPickle.load(f)


questionIndexes = np.random.choice(a=range(201, 893), size=60)

sentences = uberDict
print len(sentences)

real_indexes = []
for index in questionIndexes:
    if index in patternDict and index in questionDict:
        real_indexes.append(index)

#np.random.seed(42)
preprocessing = Preprocessing()
preprocessing.loadParser()
answerRetrieval = AnswerRetrieval()
answerExtraction = BaselineAnswerExtraction()

RR_list = list()
zero_one_list = list()
for questionIndex in real_indexes:
    questionString = questionDict[questionIndex]
    print questionString
    question = preprocessing.rawTextToSentences(questionString)[0]

    t1 = time()
    retrievedSentences = answerRetrieval.retrieve(question, sentences)
    for score, sentence  in retrievedSentences:
        print sentence.question_ID, questionIndex, sentence.label, sentence.__str__()

    t2 = time()

    print "Retrieve time = " + str(t2-t1)

    RR_list.append(calculateRR(questionIndex, retrievedSentences))

    t1 = time()
    answer = answerExtraction.extract(question, retrievedSentences)
    print 'Extracted answer: ' + answer
    t2 = time()

    print "Extraction time = " + str(t2-t1)

    zero_one_list.append(calculateMatch(answer + "\n", patternDict[questionIndex]))

print len(real_indexes)
print "MRR: %.3f" % (np.average(RR_list))
print "Zero one loss: %.3f" % (np.average(zero_one_list))
