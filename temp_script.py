"""lines = open("sentences_string.txt", "r").readlines()
print(len(lines))


for line in lines:
	splitted = line.split("|")
	if splitted[0].strip() == "225" and splitted[-1].strip() == "1":
		print(line)"""

# !/usr/bin/ python2
import pickle

objects = pickle.load(open("pickles/temp_pickle2.p", "rb"))

all_sentences = []
sentences_with_tags = []
count = 0
for key in objects:
    print key
    recenice = objects[key]
    for rec in recenice:
        print rec.__str__()
        print rec.get_label()
    count += 1
    if count > 20:
        break
"""
for currSent in objects:
	sentence = currSent.__str__()
	article_id = currSent.get_article_ID()
	question_id = currSent.get_question_ID()
	label = currSent.get_label()
	final = question_id + " " + article_id + " " + label + " " + sentence + "\n"
	all_sentences.append(sentence + "\n")
	sentences_with_tags.append(final)

all_file = open("sentences_string.txt", "w")
tagged_file = open("sentences_tags.txt", "w")
all_file.writelines(all_sentences)
tagged_file.writelines(sentences_with_tags)

all_file.close()
tagged_file.close()
"""
# print("aaa" == None)
