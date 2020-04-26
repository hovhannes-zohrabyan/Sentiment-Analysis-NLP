from abc import ABC
from nltk.classify import ClassifierI
from statistics import mode


class VoteClassifier(ClassifierI, ABC):

    def __init__(self, *classifiers):
        self.__classifier_list = classifiers

    def classify(self, features):
        votes = []
        for classifier in self.__classifier_list:
            vote = classifier.classify(features)
            votes.append(vote)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for classifier in self.__classifier_list:
            vote = classifier.classify(features)
            votes.append(vote)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)

        return conf
