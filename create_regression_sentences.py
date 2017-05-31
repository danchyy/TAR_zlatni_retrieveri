import cPickle as pickle
import numpy as np


judgment_lines = pickle.load(open("pickles/labeled_sentences.pickle", "rb"))
ranked_lines = open("data/ranked_list").readlines()

# map for ranked file, [(q, article_id)] -> rank
q_ranked = {}

# map for minimum scores for each question, [q_id] -> score
min_scores = {}

max_scores = {}

for ranked_line in ranked_lines:
    splitted_ranked = ranked_line.split("\t")
    q_id_ranked, rank, article_id_ranked, score = splitted_ranked[0], splitted_ranked[1], splitted_ranked[2], splitted_ranked[3]
    q_ranked[(q_id_ranked,article_id_ranked)] = (rank, score)
    if rank == "1000":
        min_scores[q_id_ranked] = score
    if rank == "1":
        max_scores[q_id_ranked] = score

sentences_regression = []
for sentence in judgment_lines:
    article_id, text, q_labels = sentence[0], sentence[1], sentence[2]
    q_ranks_scores = []
    for q_id, label in q_labels:
        if label == "1":
            rank, score = 1, float(max_scores[q_id]) + np.random.normal(0, 0.05)
        else:
            if (q_id, article_id) in q_ranked:
                rank, score = q_ranked[(q_id, article_id)]
                rank, score = int(rank), float(score)
            else:
                rank, score = 1001, 0.0
                rank, scre = 1001, float(min_scores[q_id]) - 0.15
            score += np.random.normal(0.0, 0.05)
        q_ranks_scores.append((q_id, rank, score, label))
    sentences_regression.append((article_id, text, q_ranks_scores))


pickle.dump(sentences_regression, open("pickles/sentences_regression.pickle", "wb"), protocol=2)