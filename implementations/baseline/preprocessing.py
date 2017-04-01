import spacy

from implementations.baseline.sentence import Sentence
from implementations.baseline.word import Word
from interfaces.i_preprocessing import IPreprocessing
from nltk.stem import PorterStemmer


class Preprocessing(IPreprocessing):
    def __init__(self):
        # these will be initialized in the load() function
        self.parser = None
        self.stemmer = None

    def loadParser(self):
        self.parser = spacy.load('en')
        self.stemmer = PorterStemmer()

    def loadSentences(self, filePaths, preprocessingFlags, destinationFiles):
        if type(filePaths) == str:
            filePaths = [filePaths]

        if type(preprocessingFlags) == bool:
            flagList = [preprocessingFlags for _ in filePaths]
        else:
            flagList = preprocessingFlags

        fileFlagList = zip(filePaths, flagList)

        allSentences = dict()
        for file, flag in fileFlagList:
            with open(file, "r") as f:
                fileStr = f.read().strip()
                fileStr = ' '.join(fileStr.split())

            if flag:
                sentences = self.rawTextToSentences(fileStr)
            else:
                sentences = self.processedTextToSentences(fileStr)

            allSentences[file] = sentences

        allSentencesList = list()
        for value in allSentences.values():
            allSentencesList += value

        if type(destinationFiles) == str:
            self.processedSentencesToFile(allSentencesList, destinationFiles)
        elif type(destinationFiles) == list:
            for src, dest in zip(filePaths, destinationFiles):
                if dest is None:
                    continue
                self.processedSentencesToFile(allSentences[src], dest)

        return allSentencesList

    def processedSentencesToFile(self, sentences, destinationFile):
        xconllStrings = list()

        for sentence in sentences:
            xconllStrings.append(sentence.getConllString())

        toWrite = "\n".join(xconllStrings)
        with open(destinationFile, "w") as f:
            f.write(toWrite)

    def processedTextToSentences(self, processedString):
        xconllStrings = [str.strip() for str in processedString.split('\n\n')]

        sentences = list()
        for xconllString in xconllStrings:
            sentence = Sentence.createFromConllString(xconllString)
            sentences.append(sentence)

        return sentences

    def rawTextToSentences(self, rawString):
        rawString = ' '.join(rawString.strip().split())
        doc = self.parser(unicode(rawString))
        sentences = list()
        for sent in doc.sents:
            wordList = list()

            offset = None
            for token in sent:
                if offset is None:
                    offset = token.i

                stem = self.stemmer.stem(token.text)
                word = Word(token.i - offset, token.text, token.lemma_, stem, token.tag_, token.ent_type_, token.dep_, token.head.i - offset)
                wordList.append(word)

            sentences.append(Sentence(wordList))

        return sentences
