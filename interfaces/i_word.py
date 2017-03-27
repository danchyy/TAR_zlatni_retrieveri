class IWord:
    def __init__(self):
        raise NotImplementedError

    def getToken(self):
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
