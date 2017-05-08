import cPickle
qdict = dict()
with open("qa_questions_201-893.txt", "r") as doc:
	lines = doc.readlines()
	for i in range(len(lines)):

		line = lines[i]
		if line.startswith("<top>"):
			question_index = int(lines[i+2].strip().split(": ")[1])
			question_text = lines[i+5].strip()
			qdict[question_index] = question_text

with open("questions.pickle", "wb") as f:
	cPickle.dump(qdict, f, protocol = 2)

with open("questions.pickle", "rb") as f:
	qdict = cPickle.load(f)

with open("questionsdictPickle.txt", "w") as file:
	for key in qdict.keys():
		file.write(str(key) + ":" + qdict[key]+"\n")