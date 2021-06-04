from controller.data_controller import DataRW
from model.vote_classifier import VoteClassifier
from controller.dataset_controller import DatasetController
from preprocessor.preprocess_controller import PreprocessController
from vectorizer.combined_vectorizer import CombinedVectorizer


class SentimentAnalysisModel:
    dataset_controller = DatasetController()
    preprocessor = PreprocessController()
    classifier = VoteClassifier()
    vectorizer = CombinedVectorizer.get_combined_vectorizer('tfidf')
    model = CombinedVectorizer.create_pipeline(vectorizer, classifier)

    @staticmethod
    def train(dataset, corpora, model_name='sentiment_analysis_model'):
        # dataset = self.dataset_controller.import_dataset()
        # x_train, x_test, y_train, y_test = self.preprocessor.prepare_data(dataset)
        x_train, y_train = DatasetController.get_data(dataset + '.csv')
        SentimentAnalysisModel.model.fit(x_train, y_train)
        DataRW.save_data_pickle('trained_model', model_name, SentimentAnalysisModel.model)

    @staticmethod
    def load_model(model_name='sentiment_analysis_model'):
        return DataRW.read_data_pickle('trained_model', model_name)
