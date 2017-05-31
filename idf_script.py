import cPickle as pickle
from implementations.baseline import preprocessing
import numpy as np

labeled_sentences = pickle.load(open("pickles/labeled_sentences.pickle", "rb"))
word_counter = {}
preproc = preprocessing.Preprocessing()
preproc.loadParser()

i = 0
for article_id, text, q_labels in labeled_sentences:
    for sentence in preproc.rawTextToSentences(text):
        for word in sentence.getWords():
            count = word_counter.get(word.getStem(), 0)
            word_counter[word.getStem().lower()] = count+1
    i+=1
    if i%10000 == 0:
        print i
for key in word_counter.keys():
    word_counter[key] = np.log(float(len(labeled_sentences))/float(word_counter[key]))

pickle.dump(word_counter, open("pickles/stem_idf_scores.pickle", "wb"), protocol=2)


idf_dict = pickle.load(open("pickles/stem_idf_scores.pickle", "rb"))
print idf_dict["desmond"]
print idf_dict["which"]