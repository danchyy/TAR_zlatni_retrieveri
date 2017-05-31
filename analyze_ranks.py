import cPickle as pickle

judgment_lines = open("data/qa_judgments").readlines()
ranked_lines = open("data/ranked_list").readlines()

judgment_lines = pickle.load(open("pickles/labeled_sentences.pickle", "rb"))

# map which stores (q_id, article_id) and maps to best ranked position in ranked_lines
q_article_map = {}

# map for judgment file, [(q, article_id)] -> label list
q_judgment = {}

# map for ranked file, [(q, article_id)] -> rank
q_ranked = {}


# qa_judgments interface
"""for line in judgment_lines:
    splitted = line.split(" ")
    q_id, article_id, label = splitted[0], splitted[1], splitted[2]
    if (q_id, article_id) in q_judgment:
        q_judgment[(q_id, article_id)].append(label)
    else:
        q_judgment[(q_id, article_id)] = [label]"""

# labeled_sentences interface
for line in judgment_lines:
    article_id, text, q_labels = line[0], line[1], line[2]
    for q_id, label in q_labels:
        if (q_id, article_id) in q_judgment:
            q_judgment[(q_id, article_id)].append(label)
        else:
            q_judgment[(q_id, article_id)] = [label]


count = 0
for ranked_line in ranked_lines:
    splitted_ranked = ranked_line.split("\t")
    q_id_ranked, rank, article_id_ranked = splitted_ranked[0], splitted_ranked[1], splitted_ranked[2]
    q_ranked[(q_id_ranked,article_id_ranked)] = rank


ranks = []

questions_visited = {}
for key in q_judgment:
    if "1" in q_judgment[key]:
        if key in q_ranked:
            print "Rank for question: " + key[0] + " and article_id: " + key[1] + ", which contains label=1 is: " + q_ranked[key]
            ranks.append(int(q_ranked[key]))
            quest = key[0]
            if quest in questions_visited:
                questions_visited[quest] += 1
            else:
                questions_visited[quest] = 1
        else:
            print "There is no rank for question: " + key[0] + " and article_id: " + key[1] + ", which contans label=1"
            ranks.append(1001)

ranks_dict = {}
for i in range(1, 1002):
    count = 0
    for curr_rank in ranks:
        if curr_rank == i:
            count += 1
    ranks_dict[i] = count

for key in ranks_dict:
    print "For rank " + str(key) + ", there are " + str(ranks_dict[key]) + " occurrences"

for key in questions_visited:
    print str(key) + " : " + str(questions_visited[key])

print "Mean of ranks is: " + str(sum(ranks) / float(len(ranks)))
print "Median of ranks is: " + str(sorted(ranks)[len(ranks) / 2])