import numpy as np

class NEEncoder():
    def __init__(self):

        self.questionPosTags = {"WDT", "WP", "WP$", "WRB"}
        self.questionPosTagToClass = {
            "when": "TIME",
            "where": "LOCATION",
            "who": "AGENT",
            "which": "THING",
            "what": "THING"
        }
        self.howDict = {
            "much": "QUANTITY",
            "many": "QUANTITY",
            "long": "TIME",
            "old": "TIME"
        }

        self.namedEntityTypeToQuestionClass = {
            "PERSON": "AGENT",
            "ORG": "AGENT",
            "FAC": "AGENT",
            "NORP": "AGENT",
            "DATE": "TIME",
            "TIME": "TIME",
            "LOC": "LOCATION",
            "GPE": "LOCATION",
            "MONEY": "QUANTITY",
            "PERCENT": "QUANTITY",
            "ORDINAL": "QUANTITY",
            "CARDINAL": "QUANTITY",
            "QUANTITY": "QUANTITY",
            "EVENT": "THING",
            "PRODUCT": "THING",
            "WORK_OF_ART": "THING",
            "LANGUAGE": "THING",
        }

        namedEntityDict = {}
        for i, key in enumerate(self.namedEntityTypeToQuestionClass.keys()):
            namedEntityDict[key] = i
        self.NEToDetailIndex = namedEntityDict



        self.questionTypeToInt = {
            "AGENT": np.array([1, 0, 0, 0, 0]),
            "TIME": np.array([0, 1, 0, 0, 0]),
            "LOCATION": np.array([0, 0, 1, 0, 0]),
            "QUANTITY": np.array([0, 0, 0, 1, 0]),
            "THING": np.array([0, 0, 0, 0, 1])
        }

    def classifyQuestion(self, question):
        for i, word in enumerate(question.wordList):
            if word.posTag not in self.questionPosTags:
                continue

            thisWord = word.wordText.lower()
            try:
                wordClass = self.questionPosTagToClass.get(thisWord)
            except:
                print wordClass
                wordClass = None
            if wordClass is not None:
                return wordClass

            if word.wordText.lower() == "how" and i < (len(question.wordList) - 1):
                nextWord = question.wordList[i+1].wordText.lower()
                try:
                    wordClass= self.howDict[nextWord]
                except:
                    wordClass = None
                if wordClass is not None:
                    return wordClass

        return None

    def classifySentence(self, sentence):
        neTypeSet = set()

        for word in sentence.wordList:
            if word.neType != "":
                neTypeSet.add(word.neType)

        return neTypeSet

    def encodeNECoarseVector(self, ne):
        return self.questionTypeToInt(self.namedEntityTypeToQuestionClass[ne])

    def encodeNEDetailVector(self, ne):
        vec = np.zeros(len(self.NEToDetailIndex))
        vec[self.NEToDetailIndex[ne]] = 1
        return vec

