import cPickle as pickle
import ROOT_SCRIPT

ROOT_PATH = ROOT_SCRIPT.get_root_path()

labeled_sentences = pickle.load(open(ROOT_PATH +"pickles/labeled_sentences_pattern_clean.pickle", "rb"))
qdict = {}
has_one_dict = {}

for i in range(len(labeled_sentences)):
    article_id, text, question_labels = labeled_sentences[i]
    for question_id, label in question_labels:
        if question_id not in has_one_dict and label=="1":
            has_one_dict[question_id] = True

        if question_id not in qdict:
            qdict[question_id] = [(i, text, label)]
        else:
            qdict[question_id].append((i, text, label))

for key in qdict.keys():
    if key not in has_one_dict:
        del qdict[key]

print qdict.__len__()

pickle.dump(qdict, open(ROOT_PATH + "pickles/question_labeled_sentence_dict.pickle", "wb"), protocol=2)

#qdict = pickle.load(open("../pickles/question_labeled_sentence_dict.pickle", "rb"))
# for i in range(201, 206):
#     print len(qdict[str(i)])

