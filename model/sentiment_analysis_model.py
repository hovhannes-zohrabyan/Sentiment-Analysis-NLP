from nltk.tokenize import word_tokenize
from controller.data_controller import DataRW
from controller.VoteClassifier import VoteClassifier


class SentimentAnalysis:

    def __init__(self):
        print("sentiment Analysis")
        self.dp = DataRW()
        # prepare_data
        self.documents = self.dp.read_data_pickle("data", "documents")
        self.word_features = self.dp.read_data_pickle("data", "word_features")
        self.vote_classifier = VoteClassifier(*self.load_models())

    def load_models(self):
        base_classifier = self.dp.read_data_pickle("trained_model", "base_classifier")
        mnb_classifier = self.dp.read_data_pickle("trained_model", "mnb_classifier")
        bernoulli_nb_classifier = self.dp.read_data_pickle("trained_model", "bernoulli_nb_classifier")
        logistic_regression_classifier = self.dp.read_data_pickle("trained_model", "logistic_regression_classifier")
        linear_svc_classifier = self.dp.read_data_pickle("trained_model", "linear_svc_classifier")
        sgdc_classifier = self.dp.read_data_pickle("trained_model", "sgdc_classifier")

        return [base_classifier, mnb_classifier, bernoulli_nb_classifier, logistic_regression_classifier, linear_svc_classifier, sgdc_classifier]

    def find_features(self, document):
        words = word_tokenize(document)
        features = {}
        for word in self.word_features:
            features[word] = (word in words)

        return features

    def sent_analyze(self, text):
        feats = self.find_features(text)
        return self.vote_classifier.classify(feats), self.vote_classifier.confidence(feats)

