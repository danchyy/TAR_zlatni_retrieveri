from interfaces.i_answer_extraction import IAnswerExtraction
from nltk.corpus import stopwords
from implementations.baseline.preprocessing import Preprocessing

class AdvancedAnswerExtraction(IAnswerExtraction):
    def __init__(self):
        self.preprocessing = Preprocessing()
        self.preprocessing.loadParser()
        self.stopWords = set(stopwords.words('english'))

        self.questionPosTags = {"WDT", "WP", "WP$", "WRB"}
        self.questionPosTagToClass = {
            "when": "TIME",
            "where": "LOCATION",
            "who": "AGENT",
            "which": "THING",
            "what": "THING"
        }
        self.howDict = {
            "much": "QUANTITY",
            "many": "QUANTITY",
            "long": "TIME",
            "old": "TIME"
        }

        self.whatDict = {
            "name": "AGENT",
        }

        self.namedEntityTypeToQuestionClass = {
            "PERSON": "AGENT",
            "ORG": "AGENT",
            "FAC": "AGENT",
            "NORP": "AGENT",
            "DATE": "TIME",
            "TIME": "TIME",
            "LOC": "LOCATION",
            "GPE": "LOCATION",
            "MONEY": "QUANTITY",
            "PERCENT": "QUANTITY",
            "ORDINAL": "QUANTITY",
            "CARDINAL": "QUANTITY",
            "QUANTITY": "QUANTITY",
            "EVENT": "THING",
            "PRODUCT": "THING",
            "WORK_OF_ART": "THING",
            "LANGUAGE": "THING",
            "LAW": "THING"
        }

    def get_name_entities(self, wanted_type, sentence):
        named_entities = []
        current_entity = None
        for word in sentence.wordList:
            if word.neType == "" or word.neType not in self.namedEntityTypeToQuestionClass or self.namedEntityTypeToQuestionClass[word.neType] != wanted_type:
                current_entity = None
                continue
            if current_entity is None:
                new_list = [word]
                named_entities.append(new_list)
                current_entity = self.namedEntityTypeToQuestionClass[word.neType]
            else:
                named_entities[-1].append(word)
        return named_entities

    def get_list_overlap(self, q_words, sent_words):
        return float((len(set(q_words) & set(sent_words)))) / (len(set(q_words) | set(sent_words)))

    def getGovernedWordsList(self, sentence):
        governedWordsList = [set() for _ in sentence.wordList]
        for word in sentence.wordList:
            if word.headIndex is not None:
                governedWordsList[word.headIndex].add(word)
        return governedWordsList

    def expand_environment_with_original(self, sentence, best_word_list):
        governed_words_list = self.getGovernedWordsList(sentence)
        expanded_words = set()
        for word in best_word_list:
            address = word.address
            for gov_word in governed_words_list[address]:
                if gov_word.rel == "appos" or gov_word.rel == "amod":
                    expanded_words.add(gov_word)
            if word.rel == "appos" or word.rel == "amod":
                expanded_words.add(sentence.wordList[word.headIndex])

        expanded_words = set(best_word_list)
        while True:
            changed = False
            expanded_new = set()
            for word in expanded_words:
                expanded_new.add(word)
                for gov_word in governed_words_list[word.address]:
                    if gov_word.rel == "compound" and gov_word not in expanded_words:
                        expanded_new.add(gov_word)
                        changed = True
                if word.rel == "compound" and sentence.wordList[word.headIndex] not in expanded_words:
                    expanded_new.add(sentence.wordList[word.headIndex])
                    changed = True
            expanded_words = expanded_new
            if not changed:
                break
        return sorted(list(expanded_words), key=lambda x: x.address)

    def expand_environment(self, sentence, best_word_list):
        governed_words_list = self.getGovernedWordsList(sentence)
        expanded_words = set()
        for word in best_word_list:
            address = word.address
            for gov_word in governed_words_list[address]:
                if gov_word.rel == "appos" or gov_word.rel == "amod":
                    expanded_words.add(gov_word)
            if word.rel == "appos" or word.rel == "amod":
                expanded_words.add(sentence.wordList[word.headIndex])

        while True:
            changed = False
            expanded_new = set()
            for word in expanded_words:
                expanded_new.add(word)
                for gov_word in governed_words_list[word.address]:
                    if gov_word.rel == "compound" and gov_word not in best_word_list and gov_word not in expanded_words:
                        expanded_new.add(gov_word)
                        changed = True
                if word.rel == "compound" and word not in best_word_list and sentence.wordList[word.headIndex] not in expanded_words:
                    expanded_new.add(sentence.wordList[word.headIndex])
                    changed = True
            expanded_words = expanded_new
            if not changed:
                break
        return sorted(list(expanded_words), key=lambda x: x.address)


    def do_agent(self, question, sentences):
        q_agents = []
        hasBegun = False
        for word in question.wordList:
            if word.neType == "" or word.neType not in self.namedEntityTypeToQuestionClass:
                continue
            if self.namedEntityTypeToQuestionClass[word.neType] == "AGENT":
                if not hasBegun:
                    hasBegun = True
                q_agents.append(word)
            else:
                if hasBegun:
                    break

        if len(q_agents) == 0:
            for sentence in sentences:
                named_entities = self.get_name_entities("AGENT", sentence)
                if len(named_entities) > 0:
                    return self.expand_environment_with_original(sentence, named_entities[0])

            return sentences[0].wordList

        maximums_sentence = []
        for sentence in sentences:
            curr_max = None
            max_list = None
            named_entities = self.get_name_entities("AGENT", sentence)
            for curr_word_list in named_entities:
                set_of_words = [x.wordText for x in curr_word_list]
                q_set_of_words = [x.wordText for x in q_agents]
                overlap = self.get_list_overlap(q_set_of_words, set_of_words)
                if (curr_max is None or overlap > curr_max) and overlap < 1.0:
                    curr_max = overlap
                    max_list = curr_word_list
            maximums_sentence.append((curr_max, max_list, sentence))

        overall_max = max(maximums_sentence, key=lambda x: x[0])

        if overall_max[0] == 0.0:
            sameClassSentences = filter(lambda sent: self.classMatch("AGENT", self.classifySentence(sent)),
                                        sentences)

            if len(sameClassSentences) == 0:
                return self.toString(sentences[0])

            return self.extractForClass("AGENT", sameClassSentences[0])
        return self.expand_environment(overall_max[2], overall_max[1])

    def extract(self, question, rankedRelevantSentences):
        parsed_question = self.preprocessing.rawTextToSentences(question)[0]
        sentences = []
        filtered_sentences = []

        # filtering sentences where no words from questions appear in sentence
        question_set = set(word for word in parsed_question.wordList)
        for sentence in rankedRelevantSentences:
            sentence = self.preprocessing.rawTextToSentences(sentence)[0]
            sentence_set = set(word.stem for word in sentence.wordList)
            for q_word in question_set:
                if q_word.wordText not in self.stopWords and q_word.stem in sentence_set:
                    filtered_sentences.append(sentence)
                    break

        #for sentence in filtered_sentences:
            #sentences.append(self.preprocessing.rawTextToSentences(sentence)[0])
        sentences = filtered_sentences

        questionClass = self.classifyQuestion(parsed_question)
        if questionClass == "AGENT":
            list_word_objects = self.do_agent(parsed_question, sentences)
            return " ".join([x.wordText for x in list_word_objects])
        if questionClass is None:
            return self.toString(sentences[0])

        sameClassSentences = filter(lambda sent: self.classMatch(questionClass, self.classifySentence(sent)),
                                    sentences)

        if len(sameClassSentences) == 0:
            return self.toString(sentences[0])

        list_word_objects = self.extractForClass(questionClass, sameClassSentences[0])
        return " ".join([x.wordText for x in list_word_objects])

    def classifyQuestion(self, question):
        for i, word in enumerate(question.wordList):
            if word.posTag not in self.questionPosTags:
                continue

            thisWord = word.wordText.lower()
            try:
                wordClass = self.questionPosTagToClass.get(thisWord)
            except:
                print wordClass
                wordClass = None
            if wordClass is not None:
                return wordClass

            if word.wordText.lower() == "how" and i < (len(question.wordList) - 1):
                nextWord = question.wordList[i + 1].wordText.lower()
                try:
                    wordClass = self.howDict[nextWord]
                except:
                    wordClass = None
                if wordClass is not None:
                    return wordClass

            if word.wordText.lower() == "what" and i < (len(question.wordList) - 1):
                for w in question.wordList[i + 1:]:
                    if w.wordText.lower() in self.whatDict:
                        wordClass = self.whatDict[w.wordText.lower()]
                        return wordClass

                return "THING"

        return None

    def classifySentence(self, tup):
        neTypeSet = set()

        sentence = tup
        for word in sentence.wordList:
            if word.neType != "":
                neTypeSet.add(word.neType)

        return neTypeSet

    def classMatch(self, questionClass, sentenceNETypeSet):
        for neType in sentenceNETypeSet:
            try:
                if questionClass == self.namedEntityTypeToQuestionClass[neType]:
                    return True
            except KeyError:
                print neType
                continue

        return False

    def toString(self, sentence):
        wordList = sentence.wordList

        startIndex = 0
        endIndex = len(wordList) - 1

        while wordList[startIndex].posTag == "punct":
            startIndex += 1

        while wordList[endIndex].posTag == "punct":
            endIndex -= 1

        return " ".join([word.wordText for word in wordList[startIndex:endIndex]])

    def extractForClass(self, questionClass, sentence):
        started = False
        extracted = list()
        for word in sentence.wordList:
            if word.neType == "" and started:
                break
            if word.neType == "" and not started:
                continue

            try:
                found_class = self.namedEntityTypeToQuestionClass[word.neType]
            except:
                found_class = None
            if word.neType != "" and questionClass == found_class and not started:
                started = True
                extracted.append(word)

            elif word.neType != "" and started:
                extracted.append(word)

        return extracted
        #return " ".join([word.wordText for word in extracted])

