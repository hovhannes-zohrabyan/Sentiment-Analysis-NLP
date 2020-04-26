import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from controller.data_controller import DataRW


class ModelTrain:

    def __init__(self):
        print("training started")
        self.dp = DataRW()
        self.word_features = []
        self.documents = []

    def find_features(self, document):
        words = word_tokenize(document)
        features = {}
        for word in self.word_features:
            features[word] = (word in words)

        return features

    def prepare_data(self):
        vocab_words = []
        pos_data, neg_data = self.dp.read_dataset("twitter_data_labeled", labeled=True)

        # TODO: Optimize this seciton to one loop
        for rev in pos_data.split("\n"):
            self.documents.append((rev, "pos"))
        for rev in neg_data.split("\n"):
            self.documents.append((rev, "neg"))

        # Save Data
        self.dp.save_data_pickle("data", "documents", self.documents)

        # Create Vocab
        pos_neg_words = [word_tokenize(pos_data), word_tokenize(neg_data)]
        for word_set in pos_neg_words:
            for word in word_set:
                vocab_words.append(word.lower())

        # Get Most Frequently used words
        vocab_words = nltk.FreqDist(vocab_words)

        self.word_features = list(vocab_words.keys())[:5000]

        self.dp.save_data_pickle("data", "word_features", self.word_features)

    def create_train_test_set(self):
        feature_sets = [(self.find_features(rev), category) for (rev, category) in self.documents]

        training_set = feature_sets[:10000]
        testing_set = feature_sets[10000:]

        return training_set, testing_set

    def train_models(self):
        training_set, testing_set = self.create_train_test_set()

        base_classifier = nltk.NaiveBayesClassifier.train(training_set)
        print("Original Naive Bayes Algo accuracy percent:",
              (nltk.classify.accuracy(base_classifier, testing_set)) * 100)
        base_classifier.show_most_informative_features(15)
        self.dp.save_data_pickle("trained_model", "base_classifier", base_classifier)

        mnb_classifier = SklearnClassifier(MultinomialNB())
        mnb_classifier.train(training_set)
        print("MNB_classifier accuracy percent:",
              (nltk.classify.accuracy(mnb_classifier, testing_set)) * 100)
        self.dp.save_data_pickle("trained_model", "mnb_classifier", mnb_classifier)

        bernoulli_nb_classifier = SklearnClassifier(BernoulliNB())
        bernoulli_nb_classifier.train(training_set)
        print("BernoulliNB_classifier accuracy percent:",
              (nltk.classify.accuracy(bernoulli_nb_classifier, testing_set)) * 100)
        self.dp.save_data_pickle("trained_model", "bernoulli_nb_classifier", bernoulli_nb_classifier)

        logistic_regression_classifier = SklearnClassifier(LogisticRegression())
        logistic_regression_classifier.train(training_set)
        print("LogisticRegression_classifier accuracy percent:",
              (nltk.classify.accuracy(logistic_regression_classifier, testing_set)) * 100)
        self.dp.save_data_pickle("trained_model", "logistic_regression_classifier", logistic_regression_classifier)

        linear_svc_classifier = SklearnClassifier(LinearSVC())
        linear_svc_classifier.train(training_set)
        print("LinearSVC_classifier accuracy percent:",
              (nltk.classify.accuracy(linear_svc_classifier, testing_set)) * 100)
        self.dp.save_data_pickle("trained_model", "linear_svc_classifier", linear_svc_classifier)

        ##NuSVC_classifier = SklearnClassifier(NuSVC())
        ##NuSVC_classifier.train(training_set)
        ##print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

        sgdc_classifier = SklearnClassifier(SGDClassifier())
        sgdc_classifier.train(training_set)
        print("SGDClassifier accuracy percent:", nltk.classify.accuracy(sgdc_classifier, testing_set) * 100)
        self.dp.save_data_pickle("trained_model", "sgdc_classifier", sgdc_classifier)

    def main_controller(self):
        self.prepare_data()
        self.train_models()


if __name__ == '__main__':
    train = ModelTrain()
    train.main_controller()