import cPickle

import itertools

from implementations.baseline.preprocessing import Preprocessing

import ROOT_SCRIPT

ROOT_PATH = ROOT_SCRIPT.get_root_path()

"""
---------------------------------
---------------------------------
CUTOFF_START and CUTOFF_END are the numbers of sentences that
will be thrown away from the start and the end of the list of
sentences sorted by sentence length.
"""

CUTOFF_START = 100
CUTOFF_END = 6000

"""
---------------------------------
---------------------------------
"""



### Expanding huge blocks of sentences to individual sentences
### using the parser

with open(ROOT_PATH  +"data/trec9_sents_no_dups.txt", "r") as f:
    lines = [l.strip() for l in f.readlines()]

print "load parser"
preprocessing = Preprocessing()
preprocessing.loadParser()
print "parser loaded"

newList = list()

for line in lines:
    splitted = line.split()

    prefix = " ".join(splitted[:2]).strip()
    line = " ".join(splitted[2:]).strip()

    sents = preprocessing.rawTextToSentences(line)

    for s in sents:
        newList.append(str(prefix) + " " + str(s))

with open("EXPANDED_SENTS.txt", "w") as f:
    f.writelines([newLine.strip()+"\n" for newLine in newList])

with open("EXPANDED_SENTS.txt", "r") as f:
    lines = [l.strip() for l in f.readlines()]


### Removing the
### duplicates

def func(line):
    splitted = line.split()
    return (" ".join(splitted[1:]).strip(), splitted[0].strip())

sentQuestionsDict = dict()
mapped = map(func, lines)

for key, value in mapped:
    try:
        sentQuestionsDict[key].append(value)
    except KeyError:
        l = [value]
        sentQuestionsDict[key] = l


def func2(tup):
    sentStr, qList = tup
    splitted = sentStr.split()

    articleID = splitted[0].strip()
    sent = " ".join(splitted[1:]).strip()

    return sent, articleID, qList


### Creating a list of (sentence string, articleID, questionList) triplets
### and sorting it descendingly by sentence length

l2 = map(func2, sentQuestionsDict.items())
l2 = sorted(l2, key=lambda x: len(x[0]), reverse=True)


### Writing the sentences into a file in the form
### articleID sentence
### q1,q2,...,qn

def func3(tripl):
    sent, articleID, qList = tripl

    return [articleID + " " + sent, ",".join(qList)]

toPrint = list(itertools.chain.from_iterable(map(func3, l2[CUTOFF_START:-CUTOFF_END])))
with open("SENTS_Q_IDS_" + str(CUTOFF_START) + "_" + str(CUTOFF_END) +".txt", "w") as f:
    f.writelines([l.strip() + "\n" for l in toPrint])

### Creating a list of (Sentence, questionList) tuples
### and pickling it to a file

print "start parser"
preprocessing = Preprocessing()
preprocessing.loadParser()
print "parser loaded"

def func4(tripl):
    sent, articleID, qList = tripl
    sentence = preprocessing.rawTextToSentences(sent, articleID)[0]

    return sentence, qList

l3 = map(func4, l2[CUTOFF_START:-CUTOFF_END])
print "finished l3. pickling..."

with open("SENT_QLIST_" + str(CUTOFF_START) + "_" + str(CUTOFF_END) + ".pickle", "wb") as f:
    cPickle.dump(l3, f, protocol=2)

