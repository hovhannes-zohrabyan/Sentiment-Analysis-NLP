from nltk import word_tokenize
from controller.sentiment_analysis_model import SentimentAnalysisModel


class InputAnalyzeController:
    __train_dataset = None

    def find_features(self, document):
        words = word_tokenize(document)
        features = {}
        for word in self.word_features:
            features[word] = (word in words)

        return features

    @staticmethod
    def train(dataset, corpora):
        if InputAnalyzeController.__train_dataset != dataset:
            SentimentAnalysisModel.train(dataset, corpora)
            InputAnalyzeController.__train_dataset = dataset

    @staticmethod
    def predict(text):
        analyzer = SentimentAnalysisModel.load_model()
        return analyzer.predict([text])
