import cPickle as pickle
from nltk.metrics.distance import edit_distance


from implementations.baseline.sentence import Sentence2

sentences_serialized = pickle.load(open("pickles/sentences_serialized.p", "rb"))

dictionary = {}

count = 0
for sentence in sentences_serialized:
    assert isinstance(sentence, Sentence2)
    question_ID, article_ID, text, label = sentence.article_ID, sentence.question_ID, sentence.__str__(), sentence.label
    key = (question_ID, article_ID)
    value = (text, label)
    if key not in dictionary:
        dictionary[key] = []
        dictionary[key].append(value)
    else:
        dictionary[key].append(value)
    count += 1
    if count % 1000:
        print "Kraj " + str(count)
