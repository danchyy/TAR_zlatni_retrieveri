import spacy

from implementations.baseline.sentence import Sentence
from implementations.baseline.word import Word
from interfaces.i_preprocessing import IPreprocessing
from nltk.stem import PorterStemmer
import cPickle as pickle


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

    def dump_trec9_sentences(self, file_path, pickle_name):
        """
        Loads the trec9 sentences and creates appropriate Sentence objects. Those objects are then serialized
        in the destination file which is given as an argument as name
        :param file_path: Path of trec9 file which contains sentences and their question tag as well as article tag
        :param pickle_name: Name of pickle file where the serialized list of sentence objects will be saved
        :return: 
        """
        input_file = open(file_path, "r")
        lines = input_file.readlines()
        input_file.close()
        sentences = list()
        for line in lines:
            line = line.strip()
            splitted_parts = line.split(" ", 2)
            question_ID = splitted_parts[0]
            article_ID = splitted_parts[1]
            text = splitted_parts[2]
            sentence_list = self.rawTextToSentences(text, question_ID, article_ID)
            for sentence in sentence_list:
                sentences.append(sentence)

        pickle.dump(sentences, open(pickle_name, "wb"))

    def dump_qa_judgment_sentences(self, file_path, pickle_name):
        """
        Loads the qa_judgment sentences and parses text, tags and label for each line. That data is then serialized
        in a pickle object.
        :param file_path: Path of qa_judgement file which contains sentences and their question tag as well as article tag and label
        :param pickle_name: Name of pickle file where the serialized list of sentence objects will be saved
        :return: 
        """
        input_file = open(file_path, "r")
        lines = input_file.readlines()
        input_file.close()
        texts = list()
        for line in lines:
            line = line.strip()
            splitted_parts = line.split(" ", 3)
            question_ID = splitted_parts[0]
            article_ID = splitted_parts[1]
            label = splitted_parts[2]
            text = splitted_parts[3]
            texts.append((question_ID, article_ID, text, label))

        pickle.dump(texts, open(pickle_name, "wb"))

    def load_pickle_file(self, pickle_file):
        return pickle.load(open(pickle_file, "rb"))

    def rawTextToSentences(self, rawString, question_ID=None, article_ID=None):
        """
        Parses the given document and store all sentences in it. Question ID and article ID can be given as well.
        Also, only one sentence can be given which will return one sentence at a time, so list of size 1 will be returned.
        :param rawString: Document or string given which need to be serialized
        :param question_ID: ID of the question, optional
        :param article_ID: ID of the article, optional
        :return: 
        """
        rawString = ' '.join(rawString.strip().split())
        doc = self.parser(unicode(rawString))
        sentences = list()
        for sent in doc.sents:
            wordList = list()

            offset = None
            for token in sent:
                if offset is None:
                    offset = token.i
                try:
                    stem = self.stemmer.stem(token.text)
                except (Exception):
                    print token.text
                    stem = token.text

                word = Word(token.i - offset, token.text, token.lemma_, stem, token.tag_, token.ent_type_, token.dep_, token.head.i - offset)
                wordList.append(word)

            sentences.append(Sentence(wordList, question_ID=question_ID, article_ID=article_ID))

        return sentences
