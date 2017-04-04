from implementations.baseline.preprocessing import Preprocessing
from interfaces.i_sentence import ISentence

print "create Preprocessing instance ..."
preprocessing = Preprocessing()
print "load parser ..."
preprocessing.loadParser()

print "process sentences ..."
print ""

while True:
    userInput = raw_input("Enter a sentence: ").strip()
    if userInput.lower() == "q" or userInput.lower() == "quit":
        print "Exiting ..."
        break

    sentences = preprocessing.rawTextToSentences(userInput)
    print "OUTPUT:"
    for sentence in sentences:
        print sentence.getConllString()
        print " -------- "
