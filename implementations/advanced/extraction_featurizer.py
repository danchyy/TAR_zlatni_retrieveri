import itertools
import cPickle as pickle
from collections import deque

import ROOT_SCRIPT
from implementations.baseline.preprocessing import Preprocessing
from implementations.baseline.sentence import Sentence
import numpy as np
from NE_encoder import NEEncoder
from implementations.baseline.word import Word

ROOT_PATH = ROOT_SCRIPT.get_root_path()


def getPosToCoarseDict():
    return {
        "NN": "NOUN",
        "NNS": "NOUN",
        "NNP": "NOUN",
        "NNPS": "NOUN",
        "VB": "VERB",
        "VBD": "VERB",
        "VBG": "VERB",
        "VBN": "VERB",
        "VBP": "VERB",
        "VBZ": "VERB",
        "BES": "VERB",
        "HVS": "VERB",
        "JJ": "ADJECTIVE",
        "JJR": "ADJECTIVE",
        "JJS": "ADJECTIVE",
        "AFX": "ADJECTIVE",
        "WDT": "WH-WORD",
        "WP": "WH-WORD",
        "WP$": "WH-WORD",
        "WRB": "WH-WORD",
        "RB": "ADVERB",
        "RBR": "ADVERB",
        "RBS": "ADVERB",
        "PRP": "PRONOUN",
        "PRP$": "PRONOUN",
        "CD": "CARDINAL_NUMBER",
        "CC": "OTHER",
        "DT": "OTHER",
        "EX": "OTHER",
        "FW": "OTHER",
        "IN": "OTHER",
        "LS": "OTHER",
        "MD": "OTHER",
        "PDT": "OTHER",
        "POS": "OTHER",
        "RP": "OTHER",
        "SYM": "OTHER",
        "TO": "OTHER",
        "UH": "OTHER",
        "-LRB-": "OTHER",
        "-PRB-": "OTHER",
        ",": "OTHER",
        ":": "OTHER",
        ".": "OTHER",
        "''": "OTHER",
        '""': "OTHER",
        "#": "OTHER",
        "``": "OTHER",
        "$": "OTHER",
        "ADD": "OTHER",
        "GW": "OTHER",
        "HYPH": "OTHER",
        "NFP": "OTHER",
        "NIL": "OTHER",
        "SP": "OTHER",
        "XX": "OTHER"
    }


class ExtractionFeaturizer():

    def __init__(self):

        self.idf_map = pickle.load(open(ROOT_PATH + "pickles/lemma_idf_scores.pickle", "rb"))
        self.NEEncoder = NEEncoder()


        posToCoarseDict = getPosToCoarseDict()
        self.detailedPOStoCoarse = posToCoarseDict

        posToIndexDict = {}
        for i, key in enumerate(posToCoarseDict.keys()):
            posToIndexDict[key] = i
        self.posToIndexDict = posToIndexDict

        coarseToIndexDict = {}
        for i, key in enumerate(set(posToCoarseDict.values())):
            coarseToIndexDict[key] = i
        self.coarseToIndexDict = coarseToIndexDict

        self.dependencyIndexDict = self.getDependencyIndexDict()

    def encode(self, sentence, question):
        """VRATI DEQ"""
        questionType = self.NEEncoder.classifyQuestion(question)
        questionType = self.NEEncoder.questionTypeToInt[questionType]
        keyWord = self.getKeyWord(sentence, question)
        wordDependencyVectors = self.encodeDependencies(sentence, keyWord)
        sequenceFeatures = deque()
        for word, dependencyVector in zip(sentence.wordList, wordDependencyVectors):
            detailedPOS = self.encodeDetailedPOS(word.getPOS())
            coarsePOS = self.encodeCoarsePOS(word.getPOS())
            detailedNE = self.NEEncoder.encodeNEDetailVector(word.getNEType())
            coarseNE = self.NEEncoder.encodeNECoarseVector(word.getNEType())
            sequenceFeatures.append(np.concatenate((questionType, detailedPOS, coarsePOS, detailedNE, coarseNE, dependencyVector)))

        return sequenceFeatures

    def encodeDetailedPOS(self, pos):
        vec = np.zeros(len(self.posToIndexDict))
        vec[self.posToIndexDict[pos]] = 1
        return vec

    def encodeCoarsePOS(self, pos):
        vec = np.zeros(len(self.coarseToIndexDict))
        vec[self.coarseToIndexDict[self.detailedPOStoCoarse[pos]]] = 1
        return vec



    def encodeDependencies(self, sentence, importantWord):
        governedWordsList = self.getGovernedWordsList(sentence)

        featureVectors = list()
        for word in sentence.wordList:
            featureVector = self.encodeWordDependency(word, governedWordsList, sentence, importantWord)
            featureVectors.append(featureVector)

        return featureVectors

    def encodeWordDependency(self, word, governedWordsList, sentence, importantWord):
        N_dep = len(self.dependencyIndexDict.keys())

        depVec = np.zeros(N_dep)
        depVec[self.dependencyIndexDict[word.rel]] = 1.0

        govVec = np.zeros(N_dep)
        for govWord in governedWordsList[word.address]:
            govVec[self.dependencyIndexDict[govWord.rel]] = 1.0

        if importantWord is None:
            depRelVector = np.zeros(len(self.dependencyIndexDict.keys()))
            coarsePosVector = np.zeros(len(self.coarseToIndexDict.keys()))
            np.concatenate((depRelVector, coarsePosVector, self.encodePathLength(None)))
        else:
            pathVec = self.getDependencyPath(word, importantWord, sentence)

        return np.concatenate((depVec, govVec, pathVec))

    def getDependencyPath(self, word, importantWord, sentence):
        depPath, posPath = self._findDepAndPosPath(word, importantWord, sentence.wordList)

        lenVec = self.encodePathLength(len(depPath))

        depRelVector = np.zeros(len(self.dependencyIndexDict.keys()))
        coarsePosVector = np.zeros(len(self.coarseToIndexDict.keys()))

        for depRel in depPath:
            depRelVector[self.dependencyIndexDict[depRel]] = 1.0

        for pos in posPath:
            coarsePosVector[self.coarseToIndexDict[self.detailedPOStoCoarse[pos]]] = 1.0

        return np.concatenate((depRelVector, coarsePosVector, lenVec))

    def _findDepAndPosPath(self, anchor, word, wordList):
        if word.address < anchor.address:
            temp1 = word
            temp2 = anchor
        else:
            temp1 = anchor
            temp2 = word

        commonAncestor = self._findCommonAncestor(temp1, temp2, wordList)

        if commonAncestor is None:
            return list(), list()

        path1 = list()
        path1pos = list()
        path2 = list()
        path2pos = list()

        #while not temp1.__eq__(commonAncestor):
        while temp1.address != commonAncestor.address:
            path1.append(temp1.rel)
            path1pos.append(temp1.posTag)
            temp1 = wordList[temp1.headIndex]

        #while not temp2.__eq__(commonAncestor):
        while temp2.address != commonAncestor.address:
            path2.append(temp2.rel)
            path2pos.append(temp2.posTag)
            temp2 = wordList[temp2.headIndex]

        if len(path1pos) > 0:
            path1pos.pop(0)
        if len(path2pos) > 0:
            path2pos.pop(0)

        path2.reverse()
        path2pos.reverse()

        dependencyPath = list(itertools.chain.from_iterable([path1, path2]))
        posPath = list(itertools.chain.from_iterable([path1pos, path2pos]))

        return dependencyPath, posPath

    def _findCommonAncestor(self, w1, w2, wordList):
        s1 = set()

        s1.add(w1)
        t1 = w1
        while True:
            s1.add(t1.address)
            if t1.headIndex != t1.address:
                t1 = wordList[t1.headIndex]
            else:
                break


        t2 = w2
        while True:
            if t2.address in s1:
                return t2
            try:
                t2 = wordList[t2.headIndex]
            except TypeError:
                return None

    def getGovernedWordsList(self, sentence):
        governedWordsList = [set() for _ in sentence.wordList]
        for word in sentence.wordList:
            if word.headIndex is not None:
                governedWordsList[word.headIndex].add(word)
        return governedWordsList

    def getDependencyIndexDict(self):
        depList = [
            "ROOT",
            "acl",
            "advcl",
            "advmod",
            "amod",
            "appos",
            "aux",
            "attr",
            "case",
            "cc",
            "ccomp",
            "clf",
            "compound",
            "conj",
            "cop",
            "csubj",
            "dep",
            "det",
            "discourse",
            "dislocated",
            "expl",
            "fixed",
            "flat",
            "goeswith",
            "iobj",
            "list",
            "mark",
            "nmod",
            "nsubj",
            "nummod",
            "obj",
            "obl",
            "orphan",
            "parataxis",
            "punct",
            "reparandum",
            "root",
            "vocative",
            "xcomp"
        ]

        depIndexDict = dict()
        for i, key in enumerate(depList):
            depIndexDict[key] = i

        return depIndexDict


    def getKeyWord(self, sentence, question):
        questionLemmas = set()
        for word in question.wordList:
            questionLemmas.add(word.getLemma())
        maxIDF = None
        keyWord = None
        for word in sentence.wordList:
            lemma = word.getLemma()
            if lemma in questionLemmas:
                idf = self.idf_map.get(lemma, -1)
                if idf > maxIDF:
                    keyWord = word
                    maxIDF = idf

        if maxIDF == -1 or maxIDF is None:
            return None
        return keyWord

    def encodePathLength(self, length):
        vec = np.zeros(5)
        if length is None:
            vec[0] = 1
        elif length == 0:
            vec[1] = 1
        elif length < 3:
            vec[2] = 1
        elif length < 6:
            vec[3] = 1
        else:
            vec[4] = 1
        return vec


prepro = Preprocessing()
prepro.loadParser()

sentence = prepro.rawTextToSentences("I am Archbishop Desmond Tutu.")[0]
question = prepro.rawTextToSentences("Who is Desmond Tutu?")[0]

ef = ExtractionFeaturizer()
print ef.getKeyWord(sentence, question)
encoding = ef.encode(sentence, question)
print len(encoding[0])
