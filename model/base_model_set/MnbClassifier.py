from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB

from .model_structure import BaseModelStructure


class MnbClassifier(BaseModelStructure):

    def __init__(self):
        BaseModelStructure.__init__(self, SklearnClassifier(MultinomialNB()), "MnbClassifier")


