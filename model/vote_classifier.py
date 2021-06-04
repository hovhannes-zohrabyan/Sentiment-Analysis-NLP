from abc import ABC
from nltk.classify import ClassifierI
from statistics import mode, StatisticsError

from controller.data_controller import DataRW
from controller.vote_classifier_train import VoteClassifierTrain


class VoteClassifier(ClassifierI, ABC):
    """
      VoteClassifier is class intended to combine multiple classification models and
      combine them using Voting method and calculating confidence.
      """

    __classifier_list = []

    @staticmethod
    def load_models():
        dp = DataRW()
        # naive_bayes_classifier = dp.read_data_pickle("trained_model", "naive_bayes_classifier")
        # mnb_classifier = dp.read_data_pickle("trained_model", "mnb_classifier")
        # bernoulli_nb_classifier = dp.read_data_pickle("trained_model", "bernoulli_nb_classifier")
        # logistic_regression_classifier = dp.read_data_pickle("trained_model", "logistic_regression_classifier")
        linear_svc_classifier = dp.read_data_pickle("trained_model", "linear_svc_classifier")
        # sgd_classifier = dp.read_data_pickle("trained_model", "sgd_classifier")

        # VoteClassifier.__classifier_list = [bernoulli_nb_classifier, logistic_regression_classifier,
        #                                     linear_svc_classifier, sgd_classifier]
        VoteClassifier.__classifier_list = [linear_svc_classifier]

    def predict(self, X_test):
        if not self.__classifier_list:
            self.load_models()

        votes = []
        for classifier in self.__classifier_list:
            vote = classifier.predict(X_test)
            votes.append(vote)

        results = []
        i = 0
        for sample_votes in zip(*votes):
            try:
                prediction = mode(sample_votes)
            except StatisticsError:
                prediction = votes[2][i]

            results.append(prediction)
            i += 1

        return results

    def fit(self, x_train, y_train):
        model = VoteClassifierTrain()
        model.main_controller(x_train, y_train)

    def confidence(self, features):
        votes = []
        for classifier in self.__classifier_list:
            vote = classifier.classify(features)
            votes.append(vote)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)

        return conf
