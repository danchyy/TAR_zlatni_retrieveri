import ROOT_SCRIPT
import numpy as np
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