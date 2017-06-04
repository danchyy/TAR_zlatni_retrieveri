import cPickle

import ROOT_SCRIPT

class GeneratorLen(object):

    def __init__(self, gen, length):
        self.gen = gen
        self.length = length
    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen

ROOT_PATH = ROOT_SCRIPT.get_root_path()

with open(ROOT_PATH + "pickles/EXTRACTION_question_labeled_sentence_dict.pickle") as f:
    qsDict = cPickle.load(f)


### ExtractionFeaturizer encode all sents


def gen(itemlist):
    featureLabels = map(lambda x: str(x[0]), enumerate(itemlist[0]))
    #featureLabels = [str(i) for i in xrange(len(itemlist[0]))]

    while len(itemlist) > 0:
        yield map(lambda wf: dict(zip(featureLabels, wf)), itemlist.popleft())

def genLabel(itemlist):
    while len(itemlist) > 0:
        labelList = itemlist.popleft()
        yield labelList


def adjustDatasetForCRF(X, y):
    X_generator = GeneratorLen(gen(X), len(X))
    y_generator = GeneratorLen(genLabel(y), len(y))

    return X_generator, y_generator

