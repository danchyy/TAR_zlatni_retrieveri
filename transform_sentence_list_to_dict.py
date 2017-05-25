import cPickle as pickle
from nltk.metrics.distance import edit_distance
from nltk import ngrams
from nltk.metrics.distance import jaccard_distance

"""
This script will match the sentence with labels for specific questions, based on minimal distance of 
kacan processed sentence and daniel processed sentence. In that way, we will connect his modified sentence with
labels from qa_jugments.
"""

dictionary_sentence = pickle.load(open("pickle_nam_materina.pickle", "rb"))
kacan_list = pickle.load(open("pickles/filtered_triplets.pickle", "rb"))

def my_dist(string1, string2):
    first_grams = set(ngrams(string1, 5))
    second_grams = set(ngrams(string2, 5))
    return jaccard_distance(first_grams, second_grams)


new_list = []
print "start"
for i in range(len(kacan_list)):
    curr_sent_object = kacan_list[i]
    article_id, list_sentence, question_list = curr_sent_object[0].strip(), curr_sent_object[1].strip(), curr_sent_object[2]
    modified_list = [] # list which contains tuples (question, label)
    for question in question_list:


        question = question.strip()
        # Iterate over all questions for sentence
        try:
            items = dictionary_sentence[(question.strip(), article_id)]
        except Exception:
            continue
        min_dist, min_label, min_sentence = None, None, None
        # Iterate over all sentences in dictionary under this question
        flag = False
        for sentence, label in items:
            sentence = sentence.strip()
            if sentence == list_sentence:
                min_label = label
                flag = True
                break
        if not flag:
            for sentence, label in items:
                sentence = sentence.strip()
                if len(sentence) > 670 or len(sentence) < 26:
                    continue
                #print sentence
                #distance = edit_distance(list_sentence, sentence)
                distance = my_dist(list_sentence, sentence)
                if min_dist is None or distance < min_dist:
                    # check minimal distance
                    min_label, min_dist, min_sentence = label, distance, sentence

        if min_label is None:
            continue

        modified_list.append((question, min_label))
        #print "+++++++++++++++++++++++++++++++++"
        #print 'minimal sentence: ' + min_sentence, min_label
        #print "================================="
    if i % 1000 == 0:
        print i
    if modified_list:
        new_list.append((article_id, list_sentence, modified_list))

pickle.dump(new_list, open("pickles/labeled_sentences.pickle", "wb"), 2)