from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from controller.data_controller import DataRW


class NNVectorizer(object):
    def __init__(self, max_features=2000, **kwargs):
        self.tokenizer = Tokenizer(num_words=max_features, split=' ')
        self.model_file = 'lstm_vectorizer_max_features_{}'.format(max_features)

    def fit(self, raw_documents, y=None):
        try:
            self.tokenizer = DataRW.read_data_pickle('trained_model', self.model_file)
            print('Using cached ' + self.model_file)
        except FileNotFoundError:
            self.tokenizer.fit_on_texts(raw_documents)
            DataRW.save_data_pickle('trained_model', self.model_file, self.tokenizer)

        return self

    def transform(self, raw_documents, copy=True):
        try:
            X = self.tokenizer.texts_to_sequences(raw_documents)
            X = pad_sequences(X)
            return X

        except Exception:
            print('Vectorizer is not fitted!!!')

    def fit_transform(self, raw_documents, y=None):
        self.fit(raw_documents)

        return self.transform(raw_documents)
