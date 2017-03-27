from i_sentence import ISentence

class IPreprocessing:
    def __init__(self):
        raise NotImplementedError

    def rawSentenceToConll(self, rawSentence, conllDestinationPath):
        """
        :param str rawSentence:
        :param str conllDestinationPath:
        :rtype: str
        """
        raise NotImplementedError

    def rawSentencesToConll(self, directoryPath, conllDestinationPath):
        """
        :param str directoryPath:
        :param str conllDestinationPath:
        :rtype: str
        """
        raise NotImplementedError

    def rawSentenceToObject(self, rawSentence):
        """
        :param str rawSentence:
        :rtype: ISentence
        """
        raise NotImplementedError

    def conllSentenceToObject(self, conllSentence):
        """
        :param str conllSentence:
        :rtype: ISentence
        """
        raise NotImplementedError

    def conllSentencesToObjects(self, directoryPath):
        """
        :param str directoryPath:
        :rtype: list[ISentence]
        """
        raise NotImplementedError

    def loadParser(self):
        # type: () -> None
        raise NotImplementedError
