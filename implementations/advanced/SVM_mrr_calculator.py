import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score
import cPickle as pickle
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(1)

train_data = np.load("../../data/train_data.npy")
train_data = poly.fit_transform(train_data)
train_labels = np.load("../../data/train_labels.npy")
test_labels = np.load("../../data/test_labels.npy")
test_data = np.load("../../data/test_data.npy")
test_data = poly.fit_transform(test_data)
questions = pickle.load(open("../../pickles/questions.pickle", "rb"))

mrr_help_map = pickle.load(open("../../pickles/mrr_help_map.pickle", "rb"))

labeled_sentences = pickle.load(open("../../pickles/labeled_sentences.pickle", "rb"))
print "Fitting SVM"

clf = svm.LinearSVC(C=2**(-14), class_weight={1: 5})
clf.fit(train_data, train_labels)
print "Fitted"
y = clf.decision_function(test_data)
mrr_map = {}

print len(y)

for i in range(len(y)):
    q, index = mrr_help_map[i]
    sentence = labeled_sentences[index]
    article_id, text, question_labels = sentence[0], sentence[1], sentence[2]
    target_label = None
    for q_id, l in question_labels:
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
    results = sorted(results, key=lambda x: x[2], reverse=True)
    print " "
    print key, questions[int(key)]
    curr_mrr = 0
    for i in range(min(len(results), 30)):
        print results[i][1], results[i][3], results[i][2]
        if str(results[i][3]) == "1":
            curr_mrr = 1.0 / (i + 1)
            break
    print "Mrr: " + str(curr_mrr)
    mrr_sum += curr_mrr
    print "====================="
mrr = mrr_sum / len(mrr_map.keys())

print mrr