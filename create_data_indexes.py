# !/usr/bin/ python2
# print("aaa" == None)
import cPickle as pickle
import numpy as np

sentences = pickle.load(open("pickles/labeled_sentences.pickle", "rb"))

print sentences[0]

positive_counter = {}
negative_counter = {}

for sentence in sentences:
    for question_id, label in sentence[2]:
        if label=='-1':
            count = negative_counter.get(question_id, 0)+1
            negative_counter[question_id] = count
        elif label=='1':
            count = positive_counter.get(question_id, 0)+1
            positive_counter[question_id] = count



question_ids = np.array(positive_counter.keys())

train_ids = question_ids[np.random.choice(664, 531, replace=False)]
print len(train_ids)
"""test_ids = []
for q_id in question_ids:
    if q_id in train_ids:
        continue
    test_ids.append(q_id)"""

#test_ids = np.array(test_ids)
test_ids = np.array(list(set(question_ids)-set(train_ids)))
print len(test_ids)

train_count, test_count = 0, 0
for sentence in sentences:
    for question_id, label in sentence[2]:
        if question_id in train_ids:
            train_count += 1
        elif question_id in test_ids:
            test_count += 1

print "Train count: " + str(train_count)
print "Test count: " + str(test_count)


print 'a'
print len(set(train_ids)&set(test_ids))

#np.save(open("data/train_indexes.npy", "wb"), train_ids)
#np.save(open("data/test_indexes.npy", "wb"), test_ids)