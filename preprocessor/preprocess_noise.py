import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class PreprocessController:
    @staticmethod
    def remove_noise(sent):
        return re.sub(r'[^a-zA-Z\s]', '', sent)

    @staticmethod
    def remove_stopwords(sent):
        non_stop_words = ['on', 'off', 'over', 'all', 'no', 'nor', 'not', 'too', 'very']
        stop_words = [word for word in stopwords.words('english') if word not in non_stop_words]
        return [word for word in sent if word not in stop_words]

    @staticmethod
    def process_sent(sentence, clean_up=False):
        sent = PreprocessController.remove_noise(sentence).lower()
        words = word_tokenize(sent)

        if clean_up:
            words = PreprocessController.remove_stopwords(words)

        return words
