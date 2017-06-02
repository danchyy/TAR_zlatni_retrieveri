import cPickle as pickle
import numpy as np
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(1)

train_data = np.load("../../data/regression_train_data.npy")
train_data = poly.fit_transform(train_data)
train_labels = np.load("../../data/regression_train_labels.npy")
test_labels = np.load("../../data/regression_test_labels.npy")
test_data = np.load("../../data/regression_test_data.npy")
test_data = poly.fit_transform(test_data)
regression_map = pickle.load(open("../../pickles/regression_mrr_help_map.pickle", "rb"))
regression_sentences = pickle.load(open("../../pickles/sentences_regression.pickle", "rb"))
questions = pickle.load(open("../../pickles/questions.pickle", "rb"))

print len(train_labels)
print len(train_data)

print train_labels

models = []
min_error, min_index = None, None
index = 0
for c in range(-20, -2):
    svr = svm.LinearSVR(C=2**c)
    print 'start fit'
    svr.fit(train_data, train_labels)

    print 'end fit'
    y = svr.predict(test_data)
    print mean_squared_error(y, test_labels)
    #from sklearn.metrics import zero_one_loss
    #print zero_one_loss(test_labels, y)

    # map which contains [q_id] -> [(article_id, text, score, label)]
    mrr_map = {}

    for i in range(len(y)):
        q, index = regression_map[i]
        sentence = regression_sentences[index]
        article_id, text, q_rank_score_labels = sentence[0], sentence[1], sentence[2]
        target_label = None
        for q_id, r, s, l in q_rank_score_labels:
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
        results = sorted(results, key=lambda x : x[2], reverse= True)
        #print " "
        #print key, questions[int(key)]
        curr_mrr = 0
        for i in range(min(len(results), 20)):
            #print results[i][1], results[i][3], results[i][2]
            if str(results[i][3]) == "1":
                curr_mrr = 1.0 / (i+1)
                break
        #print "Mrr: " + str(curr_mrr)
        mrr_sum += curr_mrr
        #print "====================="
    mrr = mrr_sum / len(mrr_map.keys())

    print "Mrr in the end for c = " + str(c) + " is: " + str(mrr)