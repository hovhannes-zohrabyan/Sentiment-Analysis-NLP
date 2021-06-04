import nltk
from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB

from .model_structure import BaseModelStructure


class BernoulliClassifier(BaseModelStructure):

    def __init__(self):
        BaseModelStructure.__init__(self, SklearnClassifier(BernoulliNB()), "BernoulliClassifier")


