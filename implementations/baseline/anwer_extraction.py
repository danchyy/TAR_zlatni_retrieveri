from implementations.baseline.preprocessing import Preprocessing
from implementations.baseline.sentence import Sentence
from interfaces.i_answer_extraction import IAnswerExtraction


class BaselineAnswerExtraction(IAnswerExtraction):

    def __init__(self):
        self.questionPosTags = { "WDT", "WP", "WP$", "WRB" }
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
            "FACILITY": "AGENT",
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


    def extract(self, question, rankedRelevantSentences):
        assert isinstance(question, Sentence)
        assert isinstance(rankedRelevantSentences, list)

        questionClass = self.classifyQuestion(question)

        if questionClass is None:
            return self.toString(rankedRelevantSentences[0])

        sameClassSentences = filter(lambda sent: self.classMatch(questionClass, self.classifySentence(sent)), rankedRelevantSentences)

        if len(sameClassSentences) == 0:
            return self.toString(rankedRelevantSentences[0])

        return self.extractForClass(questionClass, sameClassSentences[0])


    def classifyQuestion(self, question):
        for i, word in enumerate(question.wordList):
            if word.posTag not in self.questionPosTags:
                continue

            thisWord = word.wordText.lower()
            wordClass = self.questionPosTagToClass.get(thisWord)
            if wordClass is not None:
                return wordClass

            if word.wordText.lower() == "how" and i < (len(question.wordList) - 1):
                nextWord = question.wordList[i+1].wordText.lower()
                wordClass = self.howDict[nextWord]
                if wordClass is not None:
                    return wordClass

        return None

    def classifySentence(self, sentence):
        neTypeSet = set()

        for word in sentence.wordList:
            if word.neType != "":
                neTypeSet.add(word.neType)

        return neTypeSet

    def classMatch(self, questionClass, sentenceNETypeSet):
        for neType in sentenceNETypeSet:
            if questionClass == self.namedEntityTypeToQuestionClass[neType]:
                return True

        return False

    def toString(self, sentence):
        wordList = sentence.wordList

        startIndex = 0
        endIndex = len(wordList)-1

        while wordList[startIndex].posTag == "punct":
            startIndex += 1

        while wordList[endIndex].posTag == "punct":
            endIndex -= 1

        return " ".join([word.wordText for word in wordList[startIndex:endIndex]])

    def extractForClass(self, questionClass, sentence):
        started = False
        extracted = list()
        for word in sentence.wordList:
            if word.neType == "" and started:
                break
            if word.neType == "" and not started:
                continue

            if word.neType != "" and not started:
                started = True
                extracted.append(word)

            elif word.neType != "" and started:
                extracted.append(word)

        return " ".join([word.wordText for word in extracted])
