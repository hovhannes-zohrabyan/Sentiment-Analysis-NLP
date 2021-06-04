import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D

from controller.data_controller import DataRW


class LSTMNeuralNetwork:
    def __init__(self, max_features=2000, embed_dim=128, lstm_out=196):
        self.model_file = 'LSTM_model_max_features_{}_embed_dim_{}_lstm_out_{}'.\
                          format(max_features, embed_dim, lstm_out)

        self.max_features = max_features
        self.embed_dim = embed_dim
        self.lstm_out = lstm_out

        self.model = None

    def create_neural_network(self, input_length):
        model = Sequential()
        model.add(Embedding(self.max_features, self.embed_dim, input_length=input_length))
        model.add(SpatialDropout1D(0.4))
        model.add(LSTM(self.lstm_out, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(2, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def fit(self, x_train, y_train, epochs=7, batch_size=32, verbose=2):
        try:
            print('Using cached', self.model_file)
            self.model = DataRW.read_data_pickle('trained_model', self.model_file)
        except FileNotFoundError:
            self.model = self.create_neural_network(x_train.shape[1])
            y_train = pd.get_dummies(y_train).values
            self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)

            DataRW.save_data_pickle('trained_model', self.model_file, self.model)

        return self

    def predict(self, x_test):
        if self.model is None:
            print('Model is not fitted')
            return

        result = self.model.predict(x_test.reshape(1, x_test.shape[1]))

        return result

    def evaluate(self, x_test, y_test, batch_size=32, verbose=2):
        score, acc = self.model.evaluate(x_test, y_test, verbose=verbose, batch_size=batch_size)
        print("score: %.2f" % score)
        print("acc: %.2f" % acc)
