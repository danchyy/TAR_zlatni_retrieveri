from collections import OrderedDict
CLEANED_SENTENCES_PATH = "../data/trec9_sents_no_dups.txt"
ORIGINAL_SENTENCES_PATH = "../data/trec9_sents"
MISSING_DOCUMENT_STR = "MISSING DOCUMENT"
NO_MATCHING_LINE_STR = "NO MATCHING LINE IN DOCUMENT"
DOCUMENT_MAP_PATH = "../data/document_map.txt" # this is sorted by left part of AP890227-0267 tag.
QUESTION_MAP_PATH = "../data/question_map.txt" # this is sorted by question number
FULL_DOC_TAG_MAP_PATH = "../data/document_full_tag_map.txt" # this is sorted by full doc tag

def open_file_read_lines(file):
    """
    Opens file given as argument and reads all lines which are returned
    :param file: File which will be read
    :return: Lines from given file
    """
    cleaned_file = open(file, "r")
    lines = cleaned_file.readlines()
    cleaned_file.close()
    return lines

def open_file_write_lines(file, lines):
    """
    Opens the file under given name and writes given lines to the file
    :param file: File where lines will be written
    :param lines: Lines which will be written
    :return: None
    """
    out_file = open(file, "w")
    out_file.writelines(lines)
    out_file.close()


def clean_sentences():
    lines = open_file_read_lines(ORIGINAL_SENTENCES_PATH)
    filtered_sentences = set()

    for line in lines:
        filtered_sentences.add(line)

    open_file_write_lines(CLEANED_SENTENCES_PATH, sorted(list(filtered_sentences)))


def remove_special_chars():
    lines = open_file_read_lines(CLEANED_SENTENCES_PATH)

    cleaned_lines = []
    for line in lines:
        line = line.replace("&amp;", "&").replace("&quot;", "\"").replace("&apos;", "`")
        cleaned_lines.append(line)

    open_file_write_lines(CLEANED_SENTENCES_PATH, cleaned_lines)


def remove_missing_documents():
    """
    Removes missing documents, those documents have very specific text written only with them, MISSING DOCUMENT 
    """
    lines = open_file_read_lines(CLEANED_SENTENCES_PATH)
    missing_docs_removed_lines = []

    for line in lines:
        splitted = line.split(" ", 2)
        question, document_sent, sentence = splitted[0], splitted[1], splitted[2]
        if sentence.strip() == MISSING_DOCUMENT_STR:
            continue
        else:
            missing_docs_removed_lines.append(line)

    open_file_write_lines(CLEANED_SENTENCES_PATH, missing_docs_removed_lines)

def remove_no_matching_line():
    """
    Removes documents where the only sentence written is NO MATCHING LINE IN DOCUMENT 
    """
    lines = open_file_read_lines(CLEANED_SENTENCES_PATH)
    missing_docs_removed_lines = []

    for line in lines:
        splitted = line.split(" ", 2)
        question, document_sent, sentence = splitted[0], splitted[1], splitted[2]
        if sentence.strip() == NO_MATCHING_LINE_STR:
            continue
        else:
            missing_docs_removed_lines.append(line)

    open_file_write_lines(CLEANED_SENTENCES_PATH, missing_docs_removed_lines)

def clean():
    """
    Cleans the dataset, removes duplicate sentences, replaces special chars &\w , removes missing documents
    and removes documents where only line is NO MATCHING LINE IN DOCUMENT
    :return: 
    """
    clean_sentences()
    remove_special_chars()
    remove_missing_documents()
    remove_no_matching_line()

def extract_documents():
    """
    This method extracts data per document, if presumably documents are left part of this tag for example
    AP890227-0267. That is assumed for all documents, as such pattern occurs everywhere. 
    :return: Nothin
    """
    lines = open_file_read_lines(CLEANED_SENTENCES_PATH)

    document_map = {}

    for line in lines:
        splitted = line.split(" ", 2)
        question, document_sent, sentence = splitted[0], splitted[1], splitted[2]
        document = document_sent.split("-")[0]
        sentence_order = document_sent.split("-")[1]
        sentence = sentence.strip()

        if document not in document_map:
            document_map[document] = [(question, sentence_order, sentence)]
        else:
            document_map[document].append((question, sentence_order, sentence))

    sorted_dickt = OrderedDict(sorted(document_map.items()))
    data_list = []
    for key in sorted_dickt:
        print key + ":"
        data_list.append(key + ":\n")

        values = sorted_dickt[key]
        values = sorted(values, key=lambda x: x[0])
        for item in values:
            data_list.append("\t\t" + item[0] + " " + item[1] + " " + item[2] +"\n")
            print "\t\t" + item[0] + " " + item[1] + " " + item[2]

    open_file_write_lines(DOCUMENT_MAP_PATH, data_list)

def extract_documents_per_question():
    """
    This method extracts data per question and it assumes that documents are left part of tag (for example
    AP890227-0267).  
    :return: Nothin
    """
    lines = open_file_read_lines(CLEANED_SENTENCES_PATH)

    question_map = {}

    for line in lines:
        splitted = line.split(" ", 2)
        question, document_sent, sentence = splitted[0], splitted[1], splitted[2]
        document = document_sent.split("-")[0]
        sentence_order = document_sent.split("-")[1]
        sentence = sentence.strip()

        if question not in question_map:
            question_map[question] = [(document, sentence_order, sentence)]
        else:
            question_map[question].append((document, sentence_order, sentence))

    sorted_dickt = OrderedDict(sorted(question_map.items()))
    data_list = []
    for key in sorted_dickt:
        print key + ":"
        data_list.append(key + ":\n")

        values = sorted_dickt[key]
        values = sorted(values, key=lambda x: x[0])
        for item in values:
            data_list.append("\t\t" + item[0] + " " + item[1] + " " + item[2] + "\n")
            print "\t\t" + item[0] + " " + item[1] + " " + item[2]

    open_file_write_lines(QUESTION_MAP_PATH, data_list)

def extract_documents_per_whole_doc_tag():
    """
    This method extracts data per document, asuming that documents are both parts of previously mentioned tag
    :return: Nothin
    """
    lines = open_file_read_lines(CLEANED_SENTENCES_PATH)

    full_doc_tag = {}

    for line in lines:
        splitted = line.split(" ", 2)
        question, document_sent, sentence = splitted[0], splitted[1], splitted[2]
        sentence = sentence.strip()

        if question not in full_doc_tag:
            full_doc_tag[document_sent] = [(question, sentence)]
        else:
            full_doc_tag[document_sent].append((question, sentence))

    sorted_dickt = OrderedDict(sorted(full_doc_tag.items()))
    data_list = []
    for key in sorted_dickt:
        print key + ":"
        data_list.append(key + ":\n")

        values = sorted_dickt[key]
        values = sorted(values, key=lambda x: x[0])
        for item in values:
            data_list.append("\t\t" + item[0] + " " + item[1] + "\n")
            print "\t\t" + item[0] + " " + item[1]

    open_file_write_lines(FULL_DOC_TAG_MAP_PATH, data_list)

""" YOUR CODE HERE ;) """

#clean()
#extract_documents()