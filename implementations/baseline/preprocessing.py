import spacy

from implementations.baseline.sentence import Sentence
from implementations.baseline.word import Word
from interfaces.i_preprocessing import IPreprocessing
from nltk.stem import PorterStemmer
import cPickle as pickle
from nltk.metrics.distance import edit_distance
from time import time


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
        question_dict = dict()
        for line in lines:
            line = line.strip()
            splitted_parts = line.split(" ", 2)
            question_ID = splitted_parts[0]
            article_ID = splitted_parts[1]
            text = splitted_parts[2]
            sentence_list = self.rawTextToSentences(text, article_ID)
            for sentence in sentence_list:
                if question_ID not in question_dict:
                    question_dict[question_ID] = list()
                question_dict[question_ID].append(sentence)

        pickle.dump(question_dict, open(pickle_name, "wb"), 2)

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
        texts = dict()
        for line in lines:
            line = line.strip()
            splitted_parts = line.split(" ", 3)
            question_ID = splitted_parts[0]
            article_ID = splitted_parts[1]
            label = splitted_parts[2]
            text = splitted_parts[3]
            if (question_ID, article_ID) not in texts:
                texts[(question_ID, article_ID)] = list()
            texts[(question_ID, article_ID)].append((text, label))

        pickle.dump(texts, open(pickle_name, "wb"), 2)

    def set_labels_to_sentences(self, trec_pickle, qa_pickle):
        """
        Loads dictionary with trec9_sentences and qa_judgment sentences and stores label from qa_judgment
        whoose sentence is most similar to trec9 sentence.
        :param trec_pickle: pickle file containing dictionary with Sentence objects
        :param qa_pickle: pickle file containing qa_judgment
        :return: 
        """
        question_dict = self.load_pickle_file(trec_pickle)
        judgment_dict = self.load_pickle_file(qa_pickle)
        t1 = time()
        for key_question in question_dict:
            sentences = question_dict[key_question]
            for sentence in sentences:
                article_ID = sentence.get_article_ID()
                texts = judgment_dict[(key_question, article_ID)]
                minimal_distance, min_label = None, None
                for curr_item in texts:
                    text, label = curr_item[0], curr_item[1]
                    string_sentence = sentence.__str__()
                    distance = edit_distance(string_sentence, text)
                    if not minimal_distance:
                        minimal_distance, min_label = distance, label
                    elif distance < minimal_distance:
                        minimal_distance, min_label = distance, label
                sentence.set_label(min_label)

        t2 = time()
        print "diff = " + str(t2-t1)
        for key_question in question_dict:
            print key_question + " : "
            for sentence in question_dict[key_question]:
                if sentence.get_label():
                    print sentence.__str__() + " " + sentence.get_label()
                else:
                    continue
                    #print sentence.__str__()
            print ""
        pickle.dump(question_dict, open("temp_pickle.p", "wb"))

    def load_pickle_file(self, pickle_file):
        return pickle.load(open(pickle_file, "rb"))

    def rawTextToSentences(self, rawString, article_ID=None):
        """
        Parses the given document and store all sentences in it. Question ID and article ID can be given as well.
        Also, only one sentence can be given which will return one sentence at a time, so list of size 1 will be returned.
        :param rawString: Document or string given which need to be serialized
        :param question_ID: ID of the question, optional
        :param article_ID: ID of the article, optional
        :return: 
        """
        rawString = ' '.join(rawString.strip().split())

        try:
            doc = self.parser(unicode(rawString))
        except Exception, e:
            print "CANT PARSE " + rawString
            print e.message
            return list()

        sentences = list()
        for sent in doc.sents:
            wordList = list()

            offset = None
            for token in sent:
                if offset is None:
                    offset = token.i
                try:
                    stem = self.stemmer.stem(token.text)
                except Exception:
                    print "COULD NOT FIND STEM: " + token.text
                    stem = token.text

                word = Word(
                    token.i - offset,
                    str(token.text),
                    str(token.lemma_),
                    str(stem),
                    str(token.tag_),
                    str(token.ent_type_),
                    str(token.dep_),
                    token.head.i - offset)
                wordList.append(word)

            sentences.append(Sentence(wordList, article_ID=article_ID))

        return sentences
