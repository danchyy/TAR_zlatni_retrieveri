"""lines = open("sentences_string.txt", "r").readlines()
print(len(lines))


for line in lines:
	splitted = line.split("|")
	if splitted[0].strip() == "225" and splitted[-1].strip() == "1":
		print(line)"""

# !/usr/bin/ python2
"""

objects = pickle.load(open("pickles/temp_pickle2.p", "rb"))

all_sentences = []
sentences_with_tags = []
count = 0
for key in objects:
    print key
    recenice = objects[key]
    for rec in recenice:
        print rec.__str__()
        print rec.get_label()
    count += 1
    if count > 20:
        break
for currSent in objects:
	sentence = currSent.__str__()
	article_id = currSent.get_article_ID()
	question_id = currSent.get_question_ID()
	label = currSent.get_label()
	final = question_id + " " + article_id + " " + label + " " + sentence + "\n"
	all_sentences.append(sentence + "\n")
	sentences_with_tags.append(final)

all_file = open("sentences_string.txt", "w")
tagged_file = open("sentences_tags.txt", "w")
all_file.writelines(all_sentences)
tagged_file.writelines(sentences_with_tags)

all_file.close()
tagged_file.close()
"""
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

train_ids = question_ids[np.random.choice(664, 531)]
print len(train_ids)
test_ids = np.array(list(set(question_ids)-set(train_ids)))
"""test_ids = np.array([])
for key in positive_counter:
    if key not in train_ids:
        np.append(test_ids, key)"""

print train_ids
print test_ids

print 'a'
print len(set(train_ids)&set(test_ids))

np.save(open("data/train_indexes.npy", "wb"), train_ids)
np.save(open("data/test_indexes.npy", "wb"), test_ids)