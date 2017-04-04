from interfaces.i_word import IWord


class Word(IWord):
    @staticmethod
    def createFromConllString(conllString):
        parts = [p.strip() for p in conllString.split(" | ")]

        address = int(parts[0])
        wordText = parts[1]
        lemma = parts[2]
        stem = parts[3]
        posTag = parts[4]
        neType = parts[5]
        rel = parts[6]
        headIndex = int(parts[7])

        return Word(address, wordText, lemma, stem, posTag, neType, rel, headIndex)

    def getConllString(self):
        return " | ".join([str(self.address), self.wordText, self.lemma, self.stem, self.posTag, self.neType, self.rel, str(self.headIndex)])

    def __init__(self, address, wordText, lemma, stem, posTag, neType, rel, headIndex):
        self.address = address
        self.wordText = wordText
        self.lemma = lemma
        self.stem = stem
        self.posTag = posTag
        self.neType = neType
        self.rel = rel
        self.headIndex = headIndex

    def getWordText(self):
        return self.wordText

    def getLemma(self):
        return self.lemma

    def getStem(self):
        return self.stem

    def getPOS(self):
        return self.posTag

    def getNEType(self):
        return self.neType

    def getDependencyRelation(self):
        return self.rel

    def getHeadIndex(self):
        return self.headIndex

    def __str__(self):
        return self.getConllString()
