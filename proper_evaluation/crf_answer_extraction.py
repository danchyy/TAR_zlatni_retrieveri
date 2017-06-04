import cPickle
import sklearn_crfsuite
import ROOT_SCRIPT
from implementations.advanced.crf_dataset_adjust import adjustDatasetForCRF

ROOT_PATH = ROOT_SCRIPT.get_root_path()

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c2=0.1,
    max_iterations=200,
    all_possible_transitions=True,
    verbose=True,
    num_memories=1
)

with open(ROOT_PATH + "extraction_X.pickle", "rb") as f:
    X = cPickle.load(f)

with open(ROOT_PATH + "extraction_y.pickle", "rb") as f:
    y = cPickle.load(f)

N_train = int(0.7 * len(X))

X_train = X[:N_train]
y_train = y[:N_train]

X_test = X[N_train:]
y_test = y[N_train:]

X = None
y = None

X_train_generator, y_train_generator = adjustDatasetForCRF(X_train, y_train)

crf.fit(X_train_generator, y_train_generator)

X_train_generator = None
y_train_generator = None


X_test_generator, y_test_generator = adjustDatasetForCRF(X_test, y_test)

y_pred = crf.predict(X_test_generator)
X_test_generator = None

TP = 0
TN = 0
FP = 0
FN = 0


with open(ROOT_PATH + "EXTRACTION_question_labeled_sentence_dict.pickle", "rb") as f:
    extractionDict = cPickle.load(f)

for i, (pred_seq, true_seq) in enumerate(zip(y_pred, y_test_generator)):
