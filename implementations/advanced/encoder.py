from gensim.models import KeyedVectors
import cPickle as pickle
import numpy as np
from implementations.baseline.preprocessing import Preprocessing
from scipy import spatial

from nltk import ngrams
from nltk.corpus import stopwords
from nltk.metrics.distance import jaccard_distance

from implementations.baseline.sentence import Sentence
from implementations.baseline.word import Word

import ROOT_SCRIPT

ROOT_PATH = ROOT_SCRIPT.get_root_path()

WORD_2_VEC_PATH = ROOT_PATH + 'googleWord2Vec.bin'
SENTENCES_PATH = ROOT_PATH + 'pickles/labeled_sentences.pickle'
QUESTIONS_PATH = ROOT_PATH + 'pickles/questions.pickle'

OBJECT_STRING = "obj"
SUBJECT_STRING = "subj"



class Encoder():

    def __init__(self, word2vec_path=WORD_2_VEC_PATH, sent_path=SENTENCES_PATH, q_path=QUESTIONS_PATH):
        self.word_vectors = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        self.labeled_sentences = pickle.load(open(sent_path, "rb"))
        self.questions = pickle.load(open(q_path, "rb"))
        self.token_idf_map = pickle.load(open(ROOT_PATH + "pickles/token_idf_scores.pickle", "rb"))
        self.lemma_idf_map = pickle.load(open(ROOT_PATH + "pickles/lemma_idf_scores.pickle", "rb"))
        self.bigram_idf_map = pickle.load(open(ROOT_PATH + "pickles/bigram_idf_scores.pickle", "rb"))
        self.preprocessing = Preprocessing()
        self.preprocessing.loadParser()
        self.parsed_questions = {}
        self.parsed_sentences = {}
        self.stop_words = set(stopwords.words('english'))

        self.train_ids = np.load(ROOT_PATH + "data/train_ids2.npy")
        self.test_ids = np.load(ROOT_PATH + "data/test_ids2.npy")

        self.questionPosTags = { "WDT", "WP", "WP$", "WRB" }
        self.questionPosTagToClass = {
            "when": "TIME",
            "where": "LOCATION",
            "who": "AGENT",
            "which": "THING",

        }
        self.howDict = {
            "much": "QUANTITY",
            "many": "QUANTITY",
            "long": "TIME",
            "old": "TIME"
        }

        self.whatDict = {
            "name": "AGENT",
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

            if word.wordText.lower() == "what" and i < (len(question.wordList) - 1):
                for w in question.wordList[i + 1:]:
                    if w.wordText.lower() in self.whatDict:
                        wordClass = self.whatDict[w.wordText.lower()]
                        return wordClass

                return "THING"

        return None

    def classifySentence(self, sentence):
        neTypeSet = set()

        for word in sentence.wordList:
            if word.neType != "":
                neTypeSet.add(word.neType)

        return neTypeSet

    def encode_length(self, sentence):
        length = 0
        for word in sentence.wordList:
            if word.rel != "punct":
                length += 1

        if length <= 5:
            return np.array([1.0, 0.0, 0.0, 0.0])
        elif length <= 20:
            return np.array([0.0, 1.0, 0.0, 0.0])
        elif length <= 35:
            return np.array([0.0, 0.0, 1.0, 0.0])
        return np.array([0.0, 0.0, 0.0, 1.0])

    def encode_question_length(self, question):
        length = 0
        for word in question.wordList:
            if word.rel != "punct":
                length += 1

        if length <= 4:
            return np.array([1.0, 0.0, 0.0, 0.0])
        elif length <= 8:
            return np.array([0.0, 1.0, 0.0, 0.0])
        elif length <= 14:
            return np.array([0.0, 0.0, 1.0, 0.0])
        return np.array([0.0, 0.0, 0.0, 1.0])

    def my_dist(self, string1, string2):
        first_grams = set(ngrams(string1, 3))
        second_grams = set(ngrams(string2, 3))
        return jaccard_distance(first_grams, second_grams)

    def encode(self, q_id, sent_index):
        question_data = self.parsed_questions[q_id]
        sentence_data = self.parsed_sentences[sent_index]
        question_text = self.questions[int(q_id)]
        sentence_text = self.labeled_sentences[sent_index][1]
        parsed_q = question_data[0]
        parsed_sent = sentence_data[0]
        word2vec_q = question_data[1]
        word2vec_sent = sentence_data[1]
        question_type = question_data[2]
        sentence_type = sentence_data[2]

        result_type = np.bitwise_and(question_type,sentence_type)
        similarity = spatial.distance.cosine(word2vec_q, word2vec_sent)
        if np.isnan(similarity):
            similarity = 0.0
        else:
            similarity = 1.0 - similarity

        assert isinstance(parsed_q, Sentence)
        assert isinstance(parsed_sent, Sentence)
        question_words = parsed_q.wordList
        sentence_words = parsed_sent.wordList

        jaccard_similarity = 1.0 - self.my_dist(question_text, sentence_text)

        overlap = 0
        bigram_overlap = 0
        question_lemmas, sentence_lemmas = set(), set()
        bigram_question_lemmas = set()
        bigram_sentence_lemmas = set()

        for i in range(1, len(question_words)):
            curr_word, previous_word = question_words[i], question_words[i-1]
            assert isinstance(curr_word, Word)
            assert isinstance(previous_word, Word)
            if i == 1:
                question_lemmas.add(previous_word.lemma)

            question_lemmas.add(curr_word.lemma)
            bigram_question_lemmas.add((previous_word.lemma, curr_word.lemma))

        for i in range(1, len(sentence_words)):
            curr_word, previous_word = sentence_words[i], sentence_words[i-1]
            assert isinstance(curr_word, Word)
            assert isinstance(previous_word, Word)
            if i == 1:
                sentence_lemmas.add(previous_word.lemma)

            sentence_lemmas.add(curr_word.lemma)
            bigram_sentence_lemmas.add((previous_word.lemma, curr_word.lemma))

        overlap_count, bigram_overlap_count = 0, 0
        for q_lemma in question_lemmas:
            if q_lemma in sentence_lemmas:
                overlap += self.lemma_idf_map.get(q_lemma, 0)
                overlap_count += 1

        for q_bigram in bigram_question_lemmas:
            if q_bigram in bigram_sentence_lemmas:
                bigram_overlap += self.bigram_idf_map.get(q_bigram, 0)
                bigram_overlap_count += 1

        sentence_length = self.encode_length(parsed_sent)
        question_length = self.encode_question_length(parsed_q)

        if bigram_overlap_count > 0:
            bigram_overlap = bigram_overlap / bigram_overlap_count
        if overlap_count > 0:
            overlap = overlap / overlap_count


        #return np.concatenate([word2vec_q, word2vec_sent, np.array([similarity]), question_type, sentence_type, np.array([overlap])])
        #return np.concatenate([np.array([similarity]), question_type, sentence_type, np.array([overlap])]) # BEST SO FAR

        return np.concatenate([np.array([similarity, jaccard_similarity, overlap, bigram_overlap]), sentence_length,
                               question_length, question_type, sentence_type])

        #return np.concatenate([np.array([similarity]), np.array([overlap])])


    def sentence2vector(self, sentence):
        vector = np.zeros(300, )
        for word in sentence.getWords():
            try:
                wordvec = self.word_vectors[word.wordText] * self.token_idf_map[word.wordText]
            except KeyError:
                wordvec = np.zeros(300, )
            vector += wordvec
        return vector

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
        for i in range(len(self.labeled_sentences)):
            parsed_sents = self.preprocessing.rawTextToSentences(self.labeled_sentences[i][1])
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
        for i in range(len(self.labeled_sentences)):
            curr_input = self.labeled_sentences[i]
            article_id, text, questions_labels = curr_input[0], curr_input[1], curr_input[2]
            for q, l in questions_labels:
                if q in self.test_ids:
                    self.test_set.append(self.encode(q, i))
                    self.test_labels.append(float(l))
                    self.map_test_index_to_real[counter] = (q, i)
                    counter += 1
                if q in self.train_ids:
                    self.train_set.append(self.encode(q, i))
                    self.train_labels.append(float(l))
            if i % 5000 == 0:
                print "Encoded sentence ad index " + str(i)

        np.save(open(ROOT_PATH + "data/train_data_no_w2v.npy", "wb"), np.array(self.train_set))
        np.save(open(ROOT_PATH + "data/train_labels_no_w2v.npy", "wb"), np.array(self.train_labels))
        np.save(open(ROOT_PATH + "data/test_data_no_w2v.npy", "wb"), np.array(self.test_set))
        np.save(open(ROOT_PATH + "data/test_labels_no_w2v.npy", "wb"), np.array(self.test_labels))
        pickle.dump(self.map_test_index_to_real, open(ROOT_PATH + "pickles/mrr_help_map_no_w2v.pickle", "wb"), protocol=2)


    def encode_all(self):
        self.create_structures()
        self.create_encoded_vectors()


#encoder = Encoder()

#print "Created encoder"
#print "Starting to create sentences and questions structures"
#encoder.create_structures()
#encoder.encode_all()
#pickle.dump(feature_vectors, open("pickles/data_pairs.pickle", "wb"), protocol=2)
#encoder.classifyQuestion(encoder.preprocessing.rawTextToSentences("What is the name he is carrying?")[0])