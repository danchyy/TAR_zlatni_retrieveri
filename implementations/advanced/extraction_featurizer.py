import ROOT_SCRIPT
from implementations.baseline.sentence import Sentence
import numpy as np

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
        "JJ": "ADJECTIVE",
        "JJR": "ADJECTIVE",
        "JJS": "ADJECTIVE",
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
        "UH": "OTHER"
    }

        self.dependencyIndexDict = self.getDependencyIndexDict()

class ExtractionFeaturizer():

    def __init__(self):
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

    def encodeDetailedPOS(self, pos):
        vec = np.zeros(len(self.posToIndexDict))
        vec[self.posToIndexDict[pos]] = 1
        return vec

    def encodeCoarsePOS(self, pos):
        vec = np.zeros(len(self.coarseToIndexDict))
        vec[self.coarseToIndexDict[self.detailedPOStoCoarse[pos]]] = 1
        return vec

    def encode(self, sentence, question):
        """VRATI DEQ"""
        #return deque[["word_features"], ["word_features"]]
        pass

    def getKeyWord(self, sentence, question):

ef = ExtractionFeaturizer()

print ef.encodeCoarsePOS("JJ")
print ef.encodeDetailedPOS("JJ")
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

        pathVec = self.getDependencyPath(word, importantWord, sentence)

        return np.concatenate((depVec, govVec, pathVec))

    def getDependencyPath(self, word, importantWord, sentence):
        depPath, posPath = self._findDepAndPosPath(word, importantWord)

        lenVec = [len(depPath)]

        depRelVector = np.zeros(len(self.dependencyIndexDict.keys()))
        coarsePosVector = np.zeros(len(self.coarseToIndexDict.keys())

        for depRel in depPath:
            depRelVector[self.dependencyIndexDict[depRel]] = 1.0

        for pos in posPath:
            coarsePosVector[self.coarseToIndexDict[self.posToCoarseDict[pos]]] = 1.0



    def _findDepAndPosPath(self, anchor, word):
        if word.index < anchor.index:
            temp1 = word
            temp2 = anchor
        else:
            temp1 = anchor
            temp2 = word

        wordList = word.parentSentence.wordList
        commonAncestor = self._findCommonAncestor(temp1, temp2)

        if commonAncestor is None:
            return list(), list()

        path1 = list()
        path1pos = list()
        path2 = list()
        path2pos = list()

        while not temp1.__eq__(commonAncestor):
            path1.append(temp1.dependencyRelation)
            path1pos.append(temp1.posTag)
            temp1 = wordList[temp1.headIndex]

        while not temp2.__eq__(commonAncestor):
            path2.append(temp2.dependencyRelation)
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

    def _findCommonAncestor(self, w1, w2):
        s1 = set()

        wordList = w1.parentSentence.wordList
        s1.add(w1)
        t1 = w1
        while True:
            s1.add(t1)
            if t1.headIndex is not None:
                t1 = wordList[t1.headIndex]
            else:
                break


        t2 = w2
        while True:
            if t2 in s1:
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
            "acl",
            "advcl",
            "advmod",
            "amod",
            "appos",
            "aux",
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
