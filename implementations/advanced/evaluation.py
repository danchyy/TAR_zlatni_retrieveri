import cPickle
from time import time

import numpy as np
import re

from implementations.advanced.answer_retrieval import AnswerRetrieval
from implementations.baseline.preprocessing import Preprocessing
from implementations.advanced.anwer_extraction import BaselineAnswerExtraction
import ROOT_SCRIPT

ROOT_PATH = ROOT_SCRIPT.get_root_path()

def calculateRR(questionIndex, retrievedSentences):
    rank = 1
    for score, sentence in retrievedSentences:
        article_id, text, q_labels = sentence[0], sentence[1], sentence[2]
        for q, l in q_labels:
            if q == str(questionIndex) and l == "1":
                return 1.0 / float(rank)
        # if sentence.question_ID == str(questionIndex) and str(sentence.label).strip() == "1":
            # return 1.0 / float(rank)

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
with open(ROOT_PATH + "pickles/labeled_sentences.pickle", "rb") as f:
    uberDict = cPickle.load(f)
t1 = time()

print "elapsed: %.3f" % (t1 - t0)

with open(ROOT_PATH + "pickles/questions.pickle", "rb") as f:
    questionDict = cPickle.load(f)

with open(ROOT_PATH + "pickles/patterns.pickle", "rb") as f:
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
    print "==============================="
    print "==============================="

    print questionIndex, questionString
    question = preprocessing.rawTextToSentences(questionString)[0]

    t1 = time()
    retrievedSentences = answerRetrieval.retrieve(question, sentences)
    for score, sentence  in retrievedSentences:
        # article_id, text, questions
        article_id, text, q_labels = sentence[0], sentence[1], sentence[2]
        label = "-1"
        question = "0000" #not correct
        for q, l in q_labels:
            if l == "1":
                label = "1"
                question = q
                break
        print article_id, sentence, label
        # print sentence[0], sentence[1], sentence[2]
#

    t2 = time()
    rr = calculateRR(questionIndex, retrievedSentences)
    RR_list.append(rr)

    print "Retrieve time = " + str(t2-t1)
    print "RR: " + str(rr)
    print "==============================="
    print "==============================="
    print "==============================="



    """t1 = time()
    answer = answerExtraction.extract(question, retrievedSentences)
    print 'Extracted answer: ' + answer
    t2 = time()

    print "Extraction time = " + str(t2-t1)

    zero_one_list.append(calculateMatch(answer + "\n", patternDict[questionIndex]))"""

print len(real_indexes)
print "MRR: %.3f" % (np.average(RR_list))
#print "Zero one loss: %.3f" % (np.average(zero_one_list))
