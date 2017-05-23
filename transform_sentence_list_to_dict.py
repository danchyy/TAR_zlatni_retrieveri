import cPickle as pickle
from nltk.metrics.distance import edit_distance

"""
This script will match the sentence with labels for specific questions, based on minimal distance of 
kacan processed sentence and daniel processed sentence. In that way, we will connect his modified sentence with
labels from qa_judgments.
"""

dictionary_sentence = pickle.load(open("pickles/temp_pickle2.p", "rb"))
kacan_list = pickle.load(open("pickles/SENT_QLIST.pickle", "rb"))

new_list = []
print "start"
for i in range(len(kacan_list)):
    curr_sent_object = kacan_list[i]
    list_sentence, question_list = curr_sent_object[0], curr_sent_object[1]
    modified_list = [] # list which contains tuples (question, label)
    print list_sentence.__str__(), len(question_list)
    for question in question_list:
        # Iterate over all questions for sentence
        sentence_objects = dictionary_sentence[question]
        min_dist, min_label = None, None
        # Iterate over all sentences in dictionary under this question
        for sentence_dict in sentence_objects:

            # if article_IDs don't match we can skip
            if sentence_dict.article_ID != list_sentence.article_ID:
                continue
            else:
                distance = edit_distance(list_sentence.__str__(), sentence_dict.__str__())
                if not min_dist or distance < min_dist:
                    # check minimal distance
                    min_label, min_dist = sentence_dict.label, distance

        modified_list.append((question, min_label))
    print i
    new_list.append((list_sentence, modified_list))

pickle.dump(new_list, open("pickles/labeled_sentences.pickle", "wb"), 2)