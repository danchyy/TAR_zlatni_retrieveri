from interfaces.i_preprocessing import IPreprocessing
from interfaces.i_answer_retrieval import IAnswerRetrieval
from interfaces.i_answer_extraction import IAnswerExtraction


class QuestionAnswering:
    def __init__(self, preprocessing, answerRetrieval, answerExtraction, srcFilePaths, doPreprocessing, destFilePaths):
        """
        :param IPreprocessing preprocessing:
        :param IAnswerRetrieval answerRetrieval:
        :param IAnswerExtraction answerExtraction:
        :param list of str srcFilePaths:
        :param lits of bool doPreprocessing:
        :param list of str destFilePaths:
        :param str filePaths:
        """

        self.preprocessing = preprocessing
        self.answerRetrieval = answerRetrieval
        self.answerExtraction = answerExtraction
        self.preprocessing.loadParser()
        self.sentences = self.preprocessing.loadSentences(srcFilePaths, doPreprocessing, destFilePaths)

    def answerQuestion(self, question):
        """
        :param str question:
        :rtype: str
        """

        questionSentence = self.preprocessing.rawTextToSentences(question)[0]
        retrievedSentences = self.answerRetrieval.retrieve(questionSentence, self.sentences)
        extractedSentence = self.answerExtraction.extract(questionSentence, retrievedSentences)

        return str(extractedSentence)

    def answerQuestions(self, questions):
        """
        :param list of str questions:
        :rtype: list of str
        """

        return [self.answerQuestion(question) for question in questions]
