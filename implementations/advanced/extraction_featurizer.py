import ROOT_SCRIPT
ROOT_PATH = ROOT_SCRIPT.get_root_path()


class ExtractionFeaturizer():

    def __init__(self):
        self.detailedPOStoCoarse = {
            "NN": "NOUN",
            "NNS": "NOUN",
            "NNP": "NOUN",
            "NNPS": "NOUN",
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


    def encode(self, sentence, question):
        """VRATI DEQ"""
        return deque[["word_features"], ["word_features"]]

