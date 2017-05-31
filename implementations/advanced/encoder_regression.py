from gensim.models import KeyedVectors
import cPickle as pickle
import numpy as np
from implementations.baseline.preprocessing import Preprocessing
from scipy import spatial

from implementations.baseline.sentence import Sentence
from implementations.baseline.word import Word

WORD_2_VEC_PATH = '../../googleWord2Vec.bin'
SENTENCES_PATH = '../../pickles/sentences_regression.pickle'
QUESTIONS_PATH = '../../pickles/questions.pickle'

class RegressionEncoder():

    def __init__(self, word2vec_path=WORD_2_VEC_PATH, sent_path=SENTENCES_PATH, q_path=QUESTIONS_PATH):
        self.word_vectors = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        self.regression_sentences = pickle.load(open(sent_path, "rb"))
        self.questions = pickle.load(open(q_path, "rb"))
        self.preprocessing = Preprocessing()
        self.preprocessing.loadParser()
        self.parsed_questions = {}
        self.parsed_sentences = {}

        self.train_ids = np.load("../../data/train_ids.npy")
        self.test_ids = np.load("../../data/test_ids.npy")

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

        self.questionTypeToInt = {
            "AGENT" : np.array([1, 0, 0, 0, 0]),
            "TIME" : np.array([0, 1, 0, 0, 0]),
            "LOCATION" : np.array([0, 0, 1, 0, 0]),
            "QUANTITY" : np.array([0, 0, 0, 1, 0]),
            "THING" : np.array([0, 0, 0, 0, 1])
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

    def classifySentence(self, sentence):
        neTypeSet = set()

        for word in sentence.wordList:
            if word.neType != "":
                neTypeSet.add(word.neType)

        return neTypeSet

    def sentence2vector(self, sentence):
        vector = np.zeros(300, )
        for word in sentence.getWords():
            try:
                wordvec = self.word_vectors[word.wordText]
            except KeyError:
                wordvec = np.zeros(300, )
            vector += wordvec
        return vector

    def encode(self, q_id, sent_index):
        question_data = self.parsed_questions[q_id]
        sentence_data = self.parsed_sentences[sent_index]
        parsed_q = question_data[0]
        parsed_sent = sentence_data[0]
        word2vec_q = question_data[1]
        word2vec_sent = sentence_data[1]
        question_type = question_data[2]
        sentence_type = sentence_data[2]
        similarity = spatial.distance.cosine(word2vec_q, word2vec_sent)
        if np.isnan(similarity):
            similarity = 0.0
        else:
            similarity = 1 - similarity

        assert isinstance(parsed_q, Sentence)
        assert isinstance(parsed_sent, Sentence)
        question_words = parsed_q.wordList
        sentence_words = parsed_sent.wordList

        overlap = 0

        question_stems, sentence_stems = set(), set()
        for q_word in question_words:
            assert isinstance(q_word, Word)
            if q_word.rel == "punct":
                continue
            question_stems.add(q_word.stem)

        for sent_word in sentence_words:
            assert isinstance(sent_word, Word)
            if sent_word.rel == "punct":
                continue
            sentence_stems.add(sent_word.stem)

        for q_stem in question_stems:
            if q_stem in sentence_stems:
                overlap += 1

        return np.concatenate([word2vec_q, word2vec_sent, np.array([similarity]), question_type, sentence_type, np.array([overlap])])

    def create_structures(self):
        print 'Starting with questions'
        for key in self.questions:
            print key
            parsed_qs = self.preprocessing.rawTextToSentences(self.questions[key])
            parsed_question = parsed_qs[0]
            question_vector = self.sentence2vector(parsed_question)
            question_class = self.classifyQuestion(parsed_question)
            if question_class is None:
                question_class_vector = self.questionTypeToInt["THING"]
            else:
                question_class_vector = self.questionTypeToInt[question_class]
            self.parsed_questions[str(key)] = (parsed_question, question_vector, question_class_vector)

        print 'Starting with sentences'
        for i in range(len(self.regression_sentences)):
            parsed_sents = self.preprocessing.rawTextToSentences(self.regression_sentences[i][1])
            parsed_sentence = parsed_sents[0]
            sentence_vector = self.sentence2vector(parsed_sentence)
            classified_sentence = self.classifySentence(parsed_sentence)
            sentence_class_vector = np.array([0, 0, 0, 0, 0])
            for ne_type in classified_sentence:
                if ne_type not in self.namedEntityTypeToQuestionClass:
                    continue
                sentence_class_vector = np.bitwise_or(sentence_class_vector, self.questionTypeToInt[self.namedEntityTypeToQuestionClass[ne_type]])
            self.parsed_sentences[i] = (parsed_sentence, sentence_vector, sentence_class_vector)
            if i % 10000 == 0:
                print "inputed sentence at index: " + str(i)

    def create_encoded_vectors(self):
        self.train_set = []
        self.test_set = []
        self.train_labels = []
        self.test_labels = []
        self.map_test_index_to_real = {}
        counter = 0
        for i in range(len(self.regression_sentences)):
            curr_input = self.regression_sentences[i]
            article_id, text, questions_ranks_scores_labels = curr_input[0], curr_input[1], curr_input[2]
            for q, rank, score, l in questions_ranks_scores_labels:
                if q in self.test_ids:
                    self.test_set.append(self.encode(q, i))
                    self.test_labels.append(score)
                    self.map_test_index_to_real[counter] = (q, i)
                    counter += 1
                if q in self.train_ids:
                    self.train_set.append(self.encode(q, i))
                    self.train_labels.append(score)
            if i % 5000 == 0:
                print "Encoded sentence ad index " + str(i)

        np.save(open("../../data/regression_train_data.npy", "wb"), np.array(self.train_set))
        np.save(open("../../data/regression_train_labels.npy", "wb"), np.array(self.train_labels))
        np.save(open("../../data/regression_test_data.npy", "wb"), np.array(self.test_set))
        np.save(open("../../data/regression_test_labels.npy", "wb"), np.array(self.test_labels))
        pickle.dump(self.map_test_index_to_real, open("../../pickles/regression_mrr_help_map.pickle", "wb"), protocol=2)

    def encode_all(self):
        self.create_structures()
        self.create_encoded_vectors()

encoder = RegressionEncoder()
print "Created encoder"
#print "Starting to create sentences and questions structures"
#encoder.create_structures()
encoder.encode_all()