from i_sentence import ISentence

class IAnswerExtraction:
    def __init__(self):
        raise NotImplementedError

    def extract(self, question, rankedRelevantSentences):
        """
        :param ISentence question:
        :param list of ISentence rankedRelevantSentences:
        :rtype: ISentence
        """
        raise NotImplementedError
