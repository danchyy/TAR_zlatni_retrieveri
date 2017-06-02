import numpy as np
from sklearn import svm
from sklearn.metrics import f1_score
import cPickle as pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
import ROOT_SCRIPT

ROOT_PATH = ROOT_SCRIPT.get_root_path()

poly = PolynomialFeatures(1)

train_data = np.load(ROOT_PATH + "data/train_data_no_w2v.npy")
train_data = poly.fit_transform(train_data)
train_labels = np.load(ROOT_PATH + "data/train_labels_no_w2v.npy")
test_labels = np.load(ROOT_PATH + "data/test_labels_no_w2v.npy")
test_data = np.load(ROOT_PATH + "data/test_data_no_w2v.npy")
test_data = poly.fit_transform(test_data)
questions = pickle.load(open(ROOT_PATH + "pickles/questions.pickle", "rb"))

mrr_help_map = pickle.load(open(ROOT_PATH + "pickles/mrr_help_map_no_w2v.pickle", "rb"))

labeled_sentences = pickle.load(open(ROOT_PATH + "pickles/labeled_sentences.pickle", "rb"))
print "Fitting SVM"

questions_mrr = set()
for key in mrr_help_map:
    questions_mrr.add(key)

print len(questions_mrr)
lines = []
for c in range(-18, -2):

    print 'MRR for C = ' + str(c) + ' and: '
    for j in [3, 5, 7]:

        clf = svm.LinearSVC(C=2**(c), class_weight={1:j})
        #clf = LogisticRegression(C=2**(c), class_weight={1: j})
        clf.fit(train_data, train_labels)
        #print "Fitted"
        y = clf.decision_function(test_data)
        mrr_map = {}

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

        #print len(mrr_map.keys())
        lines.append(str(len(mrr_map.keys())) + "\n")
        mrr_sum = 0
        number_of_zeros = 0
        for key in mrr_map:
            results = mrr_map[key]
            results = sorted(results, key=lambda x: x[2], reverse=True)
            #print " "
            lines.append("\n")
            lines.append(key + " " +  questions[int(key)] + "\n")
            #print key, questions[int(key)]
            curr_mrr = 0
            found = False
            for i in range(min(len(results), 20)):
                #print results[i][1], results[i][3], results[i][2]
                lines.append(str(results[i][1]) + " " + str(results[i][3]) + " " + str(results[i][2]) + "\n")
                if str(results[i][3]) == "1" and not found:
                    curr_mrr = 1.0 / (i + 1)
                    found = True
            if curr_mrr == 0:
                number_of_zeros += 1
            #print "Mrr: " + str(curr_mrr)
            lines.append("Mrr: " + str(curr_mrr) + "\n")
            mrr_sum += curr_mrr
            #print "====================="
            lines.append("=====================\n")
        mrr = mrr_sum / len(mrr_map.keys())

        print '\tclass_weight = ' + str(j) + ' is: ' + str(mrr),
        lines.append("For c = -5 and class weight = 3, mrr is " + str(mrr))
        print 'Number of mrr:0 = ' + str(number_of_zeros)
        #print "Mrr for c = " + str(c) + " is: " + str(mrr)

        #TOP KEK = -5, 3


#open("../../mrr_temp_file.txt", "w").writelines(lines)