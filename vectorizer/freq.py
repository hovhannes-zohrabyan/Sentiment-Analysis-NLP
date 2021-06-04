import nltk
from nltk.tokenize import sent_tokenize

from preprocessor.preprocess_noise import PreprocessController


class FreqVectorizer(object):
    """
    Argument types in functions

    raw_document is a string or file iterable.
    raw_documents is a list of strings or file iterables.
    document is a list of processed words from raw_document.
    documents is a list of lists.

    """

    pc = PreprocessController()

    def __init__(self, **kwargs):
        self.features = None

    @staticmethod
    def prepare_raw_documents(raw_documents):
        cleaned_data = []

        for text in raw_documents:
            text_sentences = sent_tokenize(text)
            processed_sentences = (FreqVectorizer.pc.process_sent(sent, clean_up=True) for sent in text_sentences)
            prepared_text = []
            for words in processed_sentences:
                prepared_text.extend(words)

            cleaned_data.append(prepared_text)

        return cleaned_data

    @staticmethod
    def get_features(documents, limit):
        all_words = []
        for word in documents:
            all_words.extend(word)

        all_words = nltk.FreqDist(all_words)
        return sorted(list(all_words.keys())[:limit])

    def create_feature_vector(self, document):
        result = []
        doc_set = set(document)
        for f in self.features:
            result.append(int(f in doc_set))

        return result

    def get_feature_vectors(self, documents):
        return [self.create_feature_vector(document) for document in documents]

    def fit(self, raw_documents, y=None, limit=4000):
        documents = self.prepare_raw_documents(raw_documents)
        features = self.get_features(documents, limit)
        self.features = features

        return self

    def transform(self, raw_documents, copy=True):
        if self.features is None:
            print('Vectorizer is not fitted!!!')
            return

        documents = self.prepare_raw_documents(raw_documents)
        X = self.get_feature_vectors(documents)

        return X

    def fit_transform(self, raw_documents, y=None):
        self.fit(raw_documents)

        return self.transform(raw_documents)
