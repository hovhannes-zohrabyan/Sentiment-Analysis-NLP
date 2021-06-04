from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

from .word2vec import Word2VecVectorizer
from .freq import FreqVectorizer
# from .nn_vectorizer import NNVectorizer


class CombinedVectorizer:
    # Parameter name_vec is a list of tuples, where each tuple consists of a vectorizer name and a vectorizer object
    # Example: [('word2vec', Word2VecVectorizer()), ('tfidf', TfidfVectorizer())]
    @staticmethod
    def create_pipeline(name_vec, clf):
        feature_union = ('feature_union', FeatureUnion(name_vec))
        pipeline = Pipeline(steps=[feature_union, ('classifier', clf)])

        return pipeline

    @staticmethod
    def get_combined_vectorizer(*names):
        vectorizers = {
            'word2vec': ('word2vec', Word2VecVectorizer()),
            'tfidf': ('tfidf', TfidfVectorizer()),
            'freq': ('freq', FreqVectorizer()),
            # 'nn': ('nn', NNVectorizer())
        }

        return [vectorizers[name] for name in names]
