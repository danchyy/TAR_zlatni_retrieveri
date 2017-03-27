from interfaces.i_preprocessing import IPreprocessing
from interfaces.i_answer_retrieval import IAnswerRetrieval
from interfaces.i_answer_extraction import IAnswerExtraction


class QuestionAnswering:
    def __init__(self, preprocessing, answerRetrieval, answerExtraction, conllDirectoryPath):
        """
        :param IPreprocessing preprocessing:
        :param IAnswerRetrieval answerRetrieval:
        :param IAnswerExtraction answerExtraction:
        :param str conllDirectoryPath:
        """

        self.preprocessing = preprocessing
        self.answerRetrieval = answerRetrieval
        self.answerExtraction = answerExtraction
        self.conllDirectoryPath = conllDirectoryPath
        self.sentences = self.preprocessing.conllSentencesToObjects(self.conllDirectoryPath)

        self.preprocessing.loadParser()

    def answerQuestion(self, question):
        """
        :param str question:
        :rtype: str
        """

        questionSentence = self.preprocessing.rawSentenceToObject(question)
        retrievedSentences = self.answerRetrieval.retrieve(questionSentence, self.sentences)
        extractedSentence = self.answerExtraction.extract(questionSentence, retrievedSentences)

        return str(extractedSentence)

    def answerQuestions(self, questions):
        """
        :param list of str questions:
        :rtype: list of str
        """

        return [self.answerQuestion(question) for question in questions]
