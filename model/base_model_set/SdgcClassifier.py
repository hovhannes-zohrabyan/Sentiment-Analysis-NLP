from nltk.classify import SklearnClassifier
from sklearn.linear_model import SGDClassifier

from .model_structure import BaseModelStructure


class SdgcClassifier(BaseModelStructure):

    def __init__(self):
        BaseModelStructure.__init__(self, SklearnClassifier(SGDClassifier()), "SdgcClassifier")



