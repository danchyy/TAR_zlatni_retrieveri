from implementations.baseline.preprocessing import Preprocessing
from implementations.baseline.sentence import Sentence
from interfaces.i_answer_extraction import IAnswerExtraction


class BaselineAnswerExtraction(IAnswerExtraction):

    def __init__(self):
        self.preprocessing = Preprocessing()
        self.preprocessing.loadParser()
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


    def extract(self, question, rankedRelevantSentences):
        question = self.preprocessing.rawTextToSentences(question)[0]
        sentences = []
        for sentence, score in rankedRelevantSentences:
            sentences.append(self.preprocessing.rawTextToSentences(sentence)[0])
        questionClass = self.classifyQuestion(question)

        if questionClass is None:
            return self.toString(sentences[0])

        sameClassSentences = filter(lambda sent: self.classMatch(questionClass, self.classifySentence(sent)), sentences)

        if len(sameClassSentences) == 0:
            return self.toString(sentences[0])

        return self.extractForClass(questionClass, sameClassSentences[0])


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

    def classifySentence(self, tup):
        neTypeSet = set()

        sentence = tup
        for word in sentence.wordList:
            if word.neType != "":
                neTypeSet.add(word.neType)

        return neTypeSet

    def classMatch(self, questionClass, sentenceNETypeSet):
        for neType in sentenceNETypeSet:
            try:
                if questionClass == self.namedEntityTypeToQuestionClass[neType]:
                    return True
            except KeyError:
                print neType
                continue

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

            try:
                found_class = self.namedEntityTypeToQuestionClass[word.neType]
            except:
                found_class = None
            if word.neType != "" and questionClass == found_class and not started:
                started = True
                extracted.append(word)

            elif word.neType != "" and started:
                extracted.append(word)

        return " ".join([word.wordText for word in extracted])
"""
preproc = Preprocessing()
preproc.loadParser()

questionString = "Who is the Greek God of the Sea?"
question = preproc.rawTextToSentences(questionString)[0]



sentenceString = "; Officials on Friday turned on the power to this island in the Aegean Sea , where Greek mythology says the god Apollo was born ."
sentence = preproc.rawTextToSentences(sentenceString)[0]

print [(word.wordText, word.neType) for word in sentence.wordList]

bae = BaselineAnswerExtraction()
print bae.extract(question, [(1.0, sentence)])"""
