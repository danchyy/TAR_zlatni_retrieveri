import cPickle as pickle
import ROOT_SCRIPT
import re
ROOT_PATH = ROOT_SCRIPT.get_root_path()

labeled_sentences = pickle.load(open(ROOT_PATH + "pickles/labeled_sentences.pickle", "rb"))
cleaned_sentences = []
with open(ROOT_PATH + "pickles/patterns.pickle", "rb") as f:
    patternDict = pickle.load(f)

for qId in patternDict.keys():
    strPatternList = patternDict[int(qId)]
    questionPatterns = map(lambda x: re.compile(x.strip(), flags=re.IGNORECASE), strPatternList)
    patternDict[str(qId)] = questionPatterns
change_counter = 0
for sentence in labeled_sentences:
    article_id, text, q_l = sentence[0], sentence[1], sentence[2]
    cleaned_q_l = []
    for q,l in q_l:
        if l == "-1":
            cleaned_q_l.append((q, l))
            continue
        questionPatterns = patternDict[q]
        res = None
        for pat in questionPatterns:
            res = re.search(pat, text)
            if res is not None:
                break

        if res is None:
            change_counter+=1
            cleaned_q_l.append((str(q), "-1"))
        else:
            cleaned_q_l.append((q, l))

    cleaned_sentences.append((article_id, text, cleaned_q_l))
print change_counter
pickle.dump(cleaned_sentences, open(ROOT_PATH + "pickles/labeled_sentences_pattern_clean.pickle", "wb"))