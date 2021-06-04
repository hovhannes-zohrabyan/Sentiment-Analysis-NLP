from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class StopWords:

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))

    def clean_sentence(self, sentence):
        words = word_tokenize(sentence)
        filtered_sentence = [w for w in words if not w in self.stop_words]
        return filtered_sentence
