from implementations.baseline.preprocessing import Preprocessing
from time import time

preprocessor = Preprocessing()
preprocessor.loadParser()

print "Dumping the TREC-9 sentences"
#preprocessor.dump_trec9_sentences(file_path="data/trec9_sents_no_dups.txt", pickle_name="data/trec9_sentences.p")

print "Dumping the qa_judments file"
#preprocessor.dump_qa_judgment_sentences(file_path="data/qa_judgments", pickle_name="data/qa_judgments.p")

preprocessor.set_labels_to_sentences("data/trec9_sentences.p", "data/qa_judgments.p")

#t0 = time()
#question_dict = preprocessor.load_pickle_file("data/trec9_sentences.p")
#t1 = time()

#print "diff = " + str(t1 - t0)