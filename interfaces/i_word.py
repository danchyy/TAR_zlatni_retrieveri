class IWord:
    def __init__(self):
        pass

    def getWordText(self):
        """
        :rtype: str
        """
        raise NotImplementedError

    def getLemma(self):
        """
        :rtype: str
        """
        raise NotImplementedError

    def getStem(self):
        """
        :rtype: str
        """
        raise NotImplementedError

    def getPOS(self):
        """
        :rtype: str
        """
        raise NotImplementedError

    def getNEType(self):
        """
        :rtype: str
        """
        raise NotImplementedError

    def getDependencyRelation(self):
        """
        :rtype: str
        """
        raise NotImplementedError

    def getHeadIndex(self):
        """
        :rtype: int
        """
        raise NotImplementedError

    @staticmethod
    def createFromConllString(conllString):
        """
        :param str conllString:
        :rtype: IWord
        """
        raise NotImplementedError

    def getConllString(self):
        """
        :rtype: str
        """
        raise NotImplementedError
