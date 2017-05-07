from implementations.baseline.word import Word
from interfaces.i_sentence import ISentence


class Sentence(ISentence):
    def __init__(self, wordList, article_ID=None, question_ID=None):
        self.article_ID = article_ID
        self.question_ID= question_ID
        if wordList is None:
            self.wordList = list()
        else:
            self.wordList = wordList

    def getWords(self):
        return self.wordList

    def getWordByAddress(self, address):
        return self.wordList[address - 1]

    def get_article_ID(self):
        return self.article_ID

    def get_question_ID(self):
        return self.question_ID

    def __str__(self):
        return " ".join([word.wordText for word in self.wordList])

    @staticmethod
    def createFromConllString(conllString):
        wordList = [Word.createFromConllString(line.strip()) for line in conllString.split("\n")]
        return Sentence(wordList)

    def getConllString(self):
        lines = list()

        for word in self.wordList:
            lines.append(word.getConllString() + "\n")

        return "".join(lines)
