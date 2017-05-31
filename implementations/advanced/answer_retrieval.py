from gensim.models.keyedvectors import KeyedVectors
from interfaces.i_answer_retrieval import IAnswerRetrieval
from scipy import spatial
import numpy as np
import heapq
from operator import itemgetter
from implementations.baseline.preprocessing import Preprocessing
from sklearn import svm
import cPickle as pickle
from time import time



class AnswerRetrieval(IAnswerRetrieval):


    def __init__(self):
        self.word_vectors = KeyedVectors.load_word2vec_format('../../googleWord2Vec.bin', binary=True)
        self.returnK  = 20

    def retrieve(self, question, sentences):
        """
        
        :param question: 
        :param sentences: List of 
        :return: 
        """
        scored_sentences = []

        question_sum = np.zeros(300, )
        for word in question.getWords():
            try:
                wordvec = self.word_vectors[word.wordText]
            except KeyError:
                wordvec = np.zeros(300, )
            question_sum += wordvec

        for sentence in sentences:
            sentence_sum = np.zeros(300, )
            # Tuple of format: article_id, text, list of tuples: ('q', 'label')
            article_id, text, question_labels = sentence[0], sentence[1], sentence[2]
            splitted_words = text.split(" ")

            for word in splitted_words:
                try:
                    wordvec = self.word_vectors[word]
                except KeyError:
                    wordvec = np.zeros(300, )

                sentence_sum += wordvec
            # scored_sentences.append((1-spatial.distance.cosine(sentence_sum, question_sum), sentence))
            score = 1-spatial.distance.cosine(sentence_sum, question_sum)
            if np.isnan(score):
                continue
            heapq.heappush(scored_sentences, (score, sentence))
            if len(scored_sentences)>self.returnK:
                heapq.heappop(scored_sentences)

        return sorted(scored_sentences, key = itemgetter(0), reverse=True)

def method():
    model = AnswerRetrieval()

    question_word = ["house"]
    sentence_words = ["flower", "building", "home", "car"]

    question = "Who was the first president of America?"
    sentences = ["The first president of America was George Washington.", "Donald Trump is the president of America.",
                 "Russian president came to Croatia.", "First dog in space was Laika."]

    prepro = Preprocessing()
    prepro.loadParser()

    sentences = prepro.rawTextToSentences(" ".join(sentences))

    question = prepro.rawTextToSentences(question)[0]

    for sentence, score in model.retrieve(question, sentences):
        print sentence, score


from sklearn.metrics import mean_squared_error

train_data = np.load("../../data/train_data.npy")
train_labels = np.load("../../data/train_labels.npy")
test_labels = np.load("../../data/test_labels.npy")
test_data = np.load("../../data/test_data.npy")
regression_map = pickle.load(open("../../pickles/regression_mrr_help_map.pickle", "rb"))
regression_sentences = pickle.load(open("../../pickles/sentences_regression.pickle", "rb"))
questions = pickle.load(open("../../pickles/questions.pickle", "rb"))

print len(train_labels)
print len(train_data)



models = []
min_error, min_index = None, None
index = 0
for c in range(-15, 1):
    svr = svm.LinearSVR(C=2**c)
    print train_labels
    print 'start fit'
    svr.fit(train_data, train_labels)

    print 'end fit'
    y = svr.predict(test_data)

    #from sklearn.metrics import zero_one_loss
    #print zero_one_loss(test_labels, y)

    # map which contains [q_id] -> [(article_id, text, score, label)]
    mrr_map = {}

    for i in range(len(y)):
        q, index = regression_map[i]
        sentence = regression_sentences[index]
        article_id, text, q_rank_score_labels = sentence[0], sentence[1], sentence[2]
        target_label = None
        for q_id, r, s, l in q_rank_score_labels:
            if q_id == q:
                target_label = l
                break

        if q not in mrr_map:
            mrr_map[q] = [(article_id, text, y[i], target_label)]
        else:
            mrr_map[q].append((article_id, text, y[i], target_label))

    mrr_sum = 0
    for key in mrr_map:
        results = mrr_map[key]
        results = sorted(results, key=lambda x : x[2], reverse= True)
        #print " "
        #print key, questions[int(key)]
        curr_mrr = 0
        for i in range(min(len(results), 20)):
            #print results[i][1], results[i][3], results[i][2]
            if str(results[i][3]) == "1":
                curr_mrr = 1.0 / (i+1)
                break
        #print "Mrr: " + str(curr_mrr)
        mrr_sum += curr_mrr
        #print "====================="
    mrr = mrr_sum / len(mrr_map.keys())

    print "Mrr in the end for c = " + str(c) + " is: " + str(mrr)
