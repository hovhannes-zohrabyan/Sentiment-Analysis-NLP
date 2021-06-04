import nltk
from nltk.classify import SklearnClassifier
from sklearn.linear_model import LogisticRegression

from .model_structure import BaseModelStructure


class LogisticRegressionClassifier(BaseModelStructure):

    def __init__(self):
        BaseModelStructure.__init__(self, SklearnClassifier(LogisticRegression()), "LogisticRegressionClassifier")


