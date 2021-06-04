from nltk.classify import SklearnClassifier
from sklearn.svm import NuSVC

from .model_structure import BaseModelStructure


class NuSvcClassifier(BaseModelStructure):

    def __init__(self):
        BaseModelStructure.__init__(self, SklearnClassifier(NuSVC()), "NuSvcClassifier")



