
data_file = open("../data/trec9_sents", "r")
lines = data_file.readlines()
data_file.close()
filtered_sentences = set()

for line in lines:
	filtered_sentences.add(line)

cleaned_file = open("../data/trec9_sents_no_dups.txt", "w")
cleaned_file.writelines(sorted(list(filtered_sentences)))
cleaned_file.close()