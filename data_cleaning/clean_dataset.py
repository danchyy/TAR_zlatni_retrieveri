
CLEANED_SENTENCES_PATH = "../data/trec9_sents_no_dups.txt"
ORIGINAL_SENTENCES_PATH = "../data/trec9_sents"
MISSING_DOCUMENT_STR = "MISSING DOCUMENT"


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


clean_sentences()
remove_special_chars()
remove_missing_documents()