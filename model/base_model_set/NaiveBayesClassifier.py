import nltk

from .model_structure import BaseModelStructure


class NaiveBayesClassifier(BaseModelStructure):

    def __init__(self):
        BaseModelStructure.__init__(self, nltk.NaiveBayesClassifier, "NaiveBayes")
