import cPickle

import re

import ROOT_SCRIPT

def getTagLabels(triplet, questionPatterns):
    _, sentStr, _ = triplet

    for pat in questionPatterns:
        res = re.search(pat, sentStr)
        if res is not None:
            break

    if res is None:
        #print "NONE"
        #print sentStr
        #print map(lambda x: x.pattern, questionPatterns)

        return ["O"] * len(sentStr.strip().split())

    start, end = res.span()

    tmpStart = start
    tmpEnd = end

    while tmpStart > 0 and sentStr[tmpStart - 1] != " ":
        tmpStart -= 1
    sentLen = len(sentStr)

    #if sentStr[tmpStart] == " ":
    #    tmpStart += 1

    while tmpEnd < (sentLen - 1) and sentStr[tmpEnd] != " ":
        tmpEnd += 1

    before = sentStr[:tmpStart].strip()
    matched = sentStr[tmpStart:tmpEnd].strip()
    after = sentStr[tmpEnd:].strip()

    #print "------------"
    #print before
    #print matched
    #print after
    #print "------------"

    tagsBefore = map(lambda x: "O", before.strip().split())
    matchedList = matched.strip().split()
    #tagBegin = ["B"]
    tagBegin = ["I"]
    tagsInside = map(lambda x: "I", matchedList[1:])
    tagsAfter = map(lambda x: "O", after.strip().split())

    return tagsBefore + tagBegin + tagsInside + tagsAfter


ROOT_PATH = ROOT_SCRIPT.get_root_path()

with open(ROOT_PATH + "pickles/question_labeled_sentence_dict.pickle", "rb") as f:
    qsDict = cPickle.load(f)

with open(ROOT_PATH + "pickles/patterns.pickle", "rb") as f:
    patternDict = cPickle.load(f)

extractionDict = dict()
for qId in qsDict.keys():
    strPatternList = patternDict[int(qId)]
    questionPatterns = map(lambda x: re.compile(x.strip(), flags=re.IGNORECASE), strPatternList)

    sentList = qsDict[qId]

    filteredSentList = list()

    for triplet in sentList:
        if triplet[2] == "1":
            filteredSentList.append((triplet[0], triplet[1], getTagLabels(triplet, questionPatterns)))

    extractionDict[qId] = filteredSentList


with open(ROOT_PATH + "pickles/EXTRACTION_question_labeled_sentence_dict.pickle", "wb") as f:
    cPickle.dump(extractionDict, f, 2)

haveB = 0
dontHaveB = 0
for key, value in extractionDict.items():
    for sentInd, sentStr, tags in value:
        flag = False
        for tag in tags:
            if tag != "O":
                flag = True
                break

        if flag:
            haveB += 1
        else:
            dontHaveB += 1

print haveB
print dontHaveB
