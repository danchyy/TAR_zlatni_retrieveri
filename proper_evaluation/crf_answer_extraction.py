
import cPickle
import numpy as np
import re
from collections import deque

import sklearn_crfsuite
import ROOT_SCRIPT
from implementations.advanced.crf_dataset_adjust import adjustDatasetForCRF


ROOT_PATH = ROOT_SCRIPT.get_root_path()

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c2=0.01,
    max_iterations=200,
    verbose=True,
)

with open(ROOT_PATH + "pickles/extraction_X.pickle", "rb") as f:
    X = cPickle.load(f)

with open(ROOT_PATH + "pickles/extraction_y.pickle", "rb") as f:
    y = cPickle.load(f)

N_train = int(0.7 * len(X))

X_train = deque(X[:N_train])
y_train = deque(y[:N_train])

X_test = deque(X[N_train:])
y_test = deque(y[N_train:])

X = None
y = None

#print len(X_train[0])
#print X_train[0]
#print len(y_train[0])
#print y_train[0]

X_train_generator, y_train_generator = adjustDatasetForCRF(X_train, y_train)


crf.fit(X_train_generator, y_train_generator)

X_train_generator = None
y_train_generator = None


X_test_generator, y_test_generator = adjustDatasetForCRF(X_test, y_test)

y_pred = crf.predict(X_test_generator)
X_test_generator = None

with open(ROOT_PATH + "pickles/EXTRACTION_question_labeled_sentence_dict.pickle", "rb") as f:
    extractionDict = cPickle.load(f)

with open(ROOT_PATH + "pickles/extraction_question_ids.pickle", "rb") as f:
    rowList = cPickle.load(f)

with open(ROOT_PATH + "pickles/patterns.pickle", "rb") as f:
    patternDict = cPickle.load(f)


def getSubstringForSequence(sentStr, pred_seq):
    wordSubList = list()

    for word, tag in zip([w.strip() for w in sentStr.split()], pred_seq):
        if tag != "O":
            wordSubList.append(word)

    return " ".join(wordSubList)


correct = 0
count = 0



for i, (pred_seq, true_seq) in enumerate(zip(y_pred, y_test_generator)):
    rowIndex = N_train + i
    qId, qsDictIndex = rowList[rowIndex]

    strPatternList = patternDict[int(qId)]
    questionPatterns = map(lambda x: re.compile(x.strip(), flags=re.IGNORECASE), strPatternList)

    sentStr = extractionDict[qId][qsDictIndex][1]
    substring = getSubstringForSequence(sentStr, pred_seq)

    found = False
    for pat in questionPatterns:
        res = re.search(pat, substring)
        if res is not None:
            found = True
            break

    if found:
        correct += 1
        print "SEQ: " + str(pred_seq)
        print "SENT: " + sentStr
        print "PATTERNS: "
        for pat in questionPatterns:
            print pat.pattern
        print "------------"
    else:
        """
        print "SEQ: " + str(pred_seq)
        print "SENT: " + sentStr
        print "PATTERNS: "
        for pat in questionPatterns:
            print pat.pattern
        print "------------"
        """
        pass

    count += 1

print correct
print count
