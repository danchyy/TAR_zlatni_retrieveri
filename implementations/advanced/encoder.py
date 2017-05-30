from gensim.models import KeyedVectors
import cPickle as pickle
import numpy as np
from implementations.baseline.preprocessing import Preprocessing

class Encoder():

    def __init__(self):
        self.word_vectors = KeyedVectors.load_word2vec_format('../../googleWord2Vec.bin', binary=True)
        self.labeled_sentences = pickle.load(open("../../pickles/labeled_sentences.pickle", "rb"))
        self.questions = pickle.load(open("../../pickles/questions.pickle", "rb"))
        self.preprocessing = Preprocessing()
        self.preprocessing.loadParser()
        self.parsed_questions = {}
        self.parsed_sentences = {}
        self.questionPosTags = { "WDT", "WP", "WP$", "WRB" }
        self.questionPosTagToClass = {
            "when": "TIME",
            "where": "LOCATION",
            "who": "AGENT",
            "which": "THING",
            "what": "THING"
        }
        self.howDict = {
            "much": "QUANTITY",
            "many": "QUANTITY",
            "long": "TIME",
            "old": "TIME"
        }

        self.namedEntityTypeToQuestionClass = {
            "PERSON": "AGENT",
            "ORG": "AGENT",
            "FAC": "AGENT",
            "NORP": "AGENT",
            "DATE": "TIME",
            "TIME": "TIME",
            "LOC": "LOCATION",
            "GPE": "LOCATION",
            "MONEY": "QUANTITY",
            "PERCENT": "QUANTITY",
            "ORDINAL": "QUANTITY",
            "CARDINAL": "QUANTITY",
            "QUANTITY": "QUANTITY",
            "EVENT": "THING",
            "PRODUCT": "THING",
            "WORK_OF_ART": "THING",
            "LANGUAGE": "THING",
        }

    def classifyQuestion(self, question):
        for i, word in enumerate(question.wordList):
            if word.posTag not in self.questionPosTags:
                continue

            thisWord = word.wordText.lower()
            try:
                wordClass = self.questionPosTagToClass.get(thisWord)
            except:
                print wordClass
                wordClass = None
            if wordClass is not None:
                return wordClass

            if word.wordText.lower() == "how" and i < (len(question.wordList) - 1):
                nextWord = question.wordList[i+1].wordText.lower()
                try:
                    wordClass= self.howDict[nextWord]
                except:
                    wordClass = None
                if wordClass is not None:
                    return wordClass

        return None

    def encode(self, sentence, question):
        sentence_vector = np.zeros(300, )
        ret_sent = self.preprocessing.rawTextToSentences(sentence)
        parsed_sentence = ret_sent[0]

        question_vector = np.zeros(300, )
        parsed_question = self.preprocessing.rawTextToSentences(question)


    def sentence2vector(self, sentence):
        vector = np.zeros(300, )
        for word in vector.getWords():
            try:
                wordvec = self.word_vectors[word.wordText]
            except KeyError:
                wordvec = np.zeros(300, )
            vector += wordvec
        return vector

    def create_sentences(self):
        for key in self.questions:
            parsed_qs = self.preprocessing.rawTextToSentences(self.questions[key])
            parsed_question = parsed_qs[0]
            question_vector = self.sentence2vector(parsed_question)
            self.parsed_questions[key] = (parsed_question, question_vector)

        for i in range(self.labeled_sentences):
            parsed_sents = self.preprocessing.rawTextToSentences(self.labeled_sentences[i][1])
            parsed_sentence = parsed_sents[0]
            sentence_vector = self.sentence2vector(parsed_sentence)
            self.parsed_sentences[i] = (parsed_sentence, sentence_vector)



