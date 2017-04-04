from i_word import IWord


class ISentence:
    def __init__(self):
        raise NotImplementedError

    def getWordByAddress(self, address):
        """
        :param int address:
        :rtype: IWord
        """
        raise NotImplementedError

    def getWords(self):
        """
        :rtype: list of IWord
        """
        raise NotImplementedError

    def getConllString(self):
        """
        :rtype: str
        """
        raise NotImplementedError

    @staticmethod
    def createFromConllString(conllString):
        """
        :param str conllString:
        :rtype: ISentence
        """
        raise NotImplementedError

    def __str__(self):
        """
        :rtype: str
        """
        raise NotImplementedError
