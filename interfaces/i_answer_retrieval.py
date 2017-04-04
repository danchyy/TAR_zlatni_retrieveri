from i_sentence import ISentence

class IAnswerRetrieval:
    def __init__(self):
        raise NotImplementedError

    def retrieve(self, question, sentences):
        """
        :param ISentence question:
        :param list of ISentence sentences:
        :rtype: list of ISentence
        """
        raise NotImplementedError
