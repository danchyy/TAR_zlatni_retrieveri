import cPickle

import itertools

from implementations.baseline.preprocessing import Preprocessing
import cPickle as pickle
import re

with open("data/qa_judgments", "r") as f:
    lines = f.readlines()



with open("pickle_nam_materina.pickle", "rb") as f:
    d = cPickle.load(f)

print len(list(itertools.chain.from_iterable([d[key] for key in d.keys()])))

"""
newList = list()
for key in d.keys():
    if key[0] == "287":
        newList.append(d[key])

lelist = list(itertools.chain.from_iterable(newList))
lelist = sorted(lelist, key=lambda x: x[0])

for el in lelist:
    print el

print len(lelist)
"""
exit()

d = dict()

parser = Preprocessing()
parser.loadParser()
pickle_regexes = pickle.load(open("pickles/patterns.pickle", "rb"))

regex_dict = {}
for key in pickle_regexes:
    regexes = pickle_regexes[key]
    pattern_list = []
    for pattern in regexes:
        prog = re.compile(pattern.strip(), flags=re.IGNORECASE)
        pattern_list.append(prog)
    regex_dict[str(key)] = pattern_list

def containsRegex(sent, regexes):
    for regexPattern in regexes:
        if re.search(regexPattern, sent) is not None:
        #if regexPattern.search(sent) is not None:
            return True

    return False


def processSentence(sentence, label, regex, parser):
    sentences = parser.rawTextToSentences(sentence)

    length = len(sentences)

    if length == 0:
        return []

    if length == 1:
        return [(str(sentences[0]), label)]

    if label == "-1":
        return [(str(sent), label) for sent in sentences]


    newList = list()
    for sent in sentences:
        if containsRegex(str(sent), regex):
            newList.append((str(sent), "1"))
        else:
            newList.append((str(sent), "-1"))

    return newList

print "START"
for i, line in enumerate(lines):
    splitted = line.split(" ", 3)
    sentence = splitted[3].strip()
    q_id, article_id, label = splitted[0], splitted[1], splitted[2]
    key = (q_id, article_id)

    #if q_id != "287":
    #    continue

    current_regexes = regex_dict[q_id]
    processedSentences = processSentence(sentence, label, current_regexes, parser)

    for sent, label in processedSentences:
        if key not in d:
            d[key] = []
        d[key].append((sent, label))

    if i % 5000 == 0:
        print i


def processSubstrings(l):
    sortedTuples = sorted(l, key=lambda x: len(x[0]), reverse=True)

    newList = list()
    for sent, label in sortedTuples:
        canEnter = True

        for otherSent, _ in newList:
            if otherSent.find(sent) != -1:
                canEnter = False
                break

        if canEnter:
            newList.append((sent, label))

    return newList


for key in d.keys():
    l = list(set(d[key]))

    onesList = filter(lambda x: x[1] == "1", l)
    minusList = filter(lambda x: x[1] == "-1", l)
    d[key] = list(itertools.chain.from_iterable([processSubstrings(onesList), processSubstrings(minusList)]))

print "STOP"

with open("pickle_nam_materina.pickle", "wb") as f:
    cPickle.dump(d, f, 2)

"""
with open("pickle_nam_materina.pickle", "rb") as f:
    d = cPickle.load(f)

print len(list(itertools.chain.from_iterable([d[key] for key in d.keys()])))

newList = list()
for key in d.keys():
    if key[0] == "287":
        newList.append(d[key])

lelist = list(itertools.chain.from_iterable(newList))
lelist = sorted(lelist, key=lambda x: x[0])

for el in lelist:
    print el

print len(lelist)
"""