from i_word import IWord


class ISentence:
    def __init__(self):
        raise NotImplementedError

    def getWords(self):
        """
        :rtype: list of IWord
        """
        raise NotImplementedError

    def getWord(self, index):
        """
        :rtype: IWord
        """
        raise NotImplementedError

    def getDependencyRelations(self):
        """
        :rtype: (str, int, int)
        """
        raise NotImplementedError

    def __str__(self):
        """
        :rtype: str
        """
        raise NotImplementedError
