from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer


# Stemming may not always return a valid word (It is a Feature not a Bug)
class StemmingController:
    """
    StemmingController is class intended to use NLTK's default stemming features with different options
    """

    def __init__(self):
        self.ps = PorterStemmer()

    def stem_words(self, words):
        result = []
        for word in words:
            result.append(self.ps.stem(word))
        return result

    def stem_sentence(self, sentence):
        word_set = word_tokenize(sentence)
        result = self.stem_words(word_set)
        return result

    def print_stemmed(self, words):
        print(self.stem_words(words))


class LemmatizeController:
    """
    StemmingController is class intended to use NLTK's default lemmatizing features with different options
    """

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def lemmatize(self, sent):
        return [self.lemmatizer.lemmatize(word) for word in sent]

