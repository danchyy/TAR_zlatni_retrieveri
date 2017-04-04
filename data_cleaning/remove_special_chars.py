import re

cleaned_file = open("../data/trec9_sents_no_dups.txt", "r")
lines = cleaned_file.readlines()
cleaned_file.close()

cleaned_lines = []
for line in lines:
	line = line.replace("&amp;", "&").replace("&quot;", "\"").replace("&apos;", "`")
	cleaned_lines.append(line)

out_file = open("../data/trec9_sents_no_dups.txt", "w")
out_file.writelines(cleaned_lines)
out_file.close()