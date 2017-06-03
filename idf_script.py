import cPickle as pickle
from implementations.baseline import preprocessing
import numpy as np

labeled_sentences = pickle.load(open("pickles/labeled_sentences.pickle", "rb"))
token_counter = {}
lemma_counter = {}
bigram_counter = {}


preproc = preprocessing.Preprocessing()
preproc.loadParser()

j = 0
for article_id, text, q_labels in labeled_sentences:
    for sentence in preproc.rawTextToSentences(text):
        words = sentence.getWords()
        occurred_token, occurred_lemma, occurred_bigram = set(), set(), set()
        for i in range(1, len(words)):
            curr_word, previous_word = words[i], words[i-1]
            previous_lemma = previous_word.getLemma()
            curr_lemma = curr_word.getLemma()

            if i == 1:
                # Token dict, we input 0th only when i = 1
                previous_token = previous_word.wordText
                previous_token_count = token_counter.get(previous_token, 0)
                token_counter[previous_token] = previous_token_count + 1
                occurred_token.add(previous_token)

                # Lemma dict, we input 0th only when i = 1
                previous_lemma_count = lemma_counter.get(previous_lemma, 0)
                lemma_counter[previous_lemma] = previous_lemma_count + 1
                occurred_lemma.add(previous_lemma)

            # Token dict, curr word
            curr_token = curr_word.wordText
            if curr_token not in occurred_token:
                curr_token_count = token_counter.get(curr_token, 0)
                token_counter[curr_token] = curr_token_count + 1
                occurred_token.add(curr_token)

            # Current lemma
            if curr_lemma not in occurred_lemma:
                curr_lemma_count = lemma_counter.get(curr_lemma, 0)
                lemma_counter[curr_lemma] = curr_lemma_count + 1
                occurred_lemma.add(curr_lemma)

            # count for bigrams
            bigram = (previous_lemma, curr_lemma)
            if bigram not in occurred_bigram:
                bigram_count = bigram_counter.get(bigram, 0)
                bigram_counter[bigram] = bigram_count + 1
                occurred_bigram.add(bigram)

    j+=1
    if j%10000 == 0:
        print j

for key in token_counter.keys():
    token_counter[key] = np.log(float(len(labeled_sentences)) / float(token_counter[key]))

for key in lemma_counter.keys():
    lemma_counter[key] = np.log(float(len(labeled_sentences)) / float(lemma_counter[key]))

for key in bigram_counter.keys():
    bigram_counter[key] = np.log(float(len(labeled_sentences)) / float(bigram_counter[key]))

pickle.dump(token_counter, open("pickles/token_idf_scores.pickle", "wb"), protocol=2)
pickle.dump(lemma_counter, open("pickles/lemma_idf_scores.pickle", "wb"), protocol=2)
pickle.dump(bigram_counter, open("pickles/bigram_idf_scores.pickle", "wb"), protocol=2)


idf_dict = pickle.load(open("pickles/token_idf_scores.pickle", "rb"))
print idf_dict["Desmond"]
print idf_dict["which"]

bigram_dict = pickle.load(open("pickles/bigram_idf_scores.pickle", "rb"))

print bigram_dict[("of", "japan")]
