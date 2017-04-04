from i_sentence import ISentence


class IPreprocessing:
    def __init__(self):
        raise NotImplementedError

    def loadParser(self):
        # type: () -> None
        raise NotImplementedError

    def loadSentences(self, filePaths, preprocessingFlags, destinationFiles):
        """
        :param list of str filePaths:
        :param list of boolean preprocessingFlags:
        :param list of str destinationFiles:
        :rtype: list of ISentence
        """
        raise NotImplementedError

    def processedSentencesToFile(self, sentences, destinationFile):
        """
        :param str destinationFile:
        :param list of ISentence sentences:
        """
        raise NotImplementedError

    def processedTextToSentences(self, xconllString):
        """
        :param str xconllString:
        :rtype: list of ISentence
        """
        raise NotImplementedError

    def rawTextToSentences(self, rawString):
        """
        :param str rawString:
        :return: list of ISentence
        """
        raise NotImplementedError
