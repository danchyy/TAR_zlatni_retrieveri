from gensim.models.keyedvectors import KeyedVectors
from interfaces.i_answer_retrieval import IAnswerRetrieval
from scipy import spatial
import numpy as np
import heapq
from operator import itemgetter
from implementations.baseline.preprocessing import Preprocessing
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