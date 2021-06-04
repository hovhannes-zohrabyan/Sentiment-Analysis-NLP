import nltk
from nltk.classify import SklearnClassifier
from sklearn.svm import LinearSVC

from .model_structure import BaseModelStructure


class LinearSvcClassifier(BaseModelStructure):

    def __init__(self):
        BaseModelStructure.__init__(self, SklearnClassifier(LinearSVC()), "LinearSvcController")



