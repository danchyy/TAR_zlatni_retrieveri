import cPickle as pickle
from time import time
from implementations.baseline.sentence import Sentence
from implementations.baseline.sentence import Sentence2


t1 = time()
objects = pickle.load(open("pickles/temp_pickle2.p","rb"))
t2 = time()

print "Diff = " + str(t2-t1)

new_sentences = []
string_format = []


for key in objects:
    curr_list = objects[key]
    t1 = time()
    for sentence in curr_list:
        article_ID = sentence.get_article_ID()
        label = sentence.get_label()
        words = sentence.getWords()
        text = ""
        for word in words:
            text += word.__str__() + " "
        text = text.strip()
        sent_obj = Sentence2(words, label=label, question_ID=key, article_ID=article_ID)
        new_sentences.append(sent_obj)
        str_sent = key + " | " + article_ID + " | " + text + " | " + label
        string_format.append(str_sent)
    t2 = time()
    print "Diff for one key: " + str(t2-t1)

pickle.dump(objects, open("pickles/sentences_serialized.p", "wb"), protocol=2)
open("sentences_string.txt", "w").writelines(string_format)
