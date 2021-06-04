import numpy as np
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class WordEmbeddingVectorizer(object):
    """
    Argument types in functions

    raw_document is a string or file iterable.
    raw_documents is a list of strings or file iterables.
    document is a list of processed words from raw_document.
    documents is a list of lists.

    """

    def __init__(self, **kwargs):
        self.model = None
        self.tokenizer = None

    # Function to create average word vector for document.
    def create_avg_feature_vector(self, document):
        feature_vec = np.zeros(self.model.wv.vector_size, dtype="float32")
        num_words = 0

        # Converting Index2Word which is a list to a set for better speed in the execution.
        index2word_set = set(self.model.wv.index2word)

        for word in document:
            if word in index2word_set:
                num_words += 1
                feature_vec = np.add(feature_vec, self.model[word])

        # Dividing the result by number of words to get average
        if num_words == 0:
            num_words = 1
        feature_vec = np.divide(feature_vec, num_words)

        return feature_vec

    # Function to create average word vector for each document.
    def get_avg_feature_vectors(self, documents, verbose):
        counter = 0
        feature_vectors = np.zeros((len(documents), self.model.wv.vector_size), dtype="float32")
        for words in documents:
            # Printing a status message every 1000th example
            if verbose and counter % 1000 == 0:
                print("Example %d of %d" % (counter, len(documents)))

            feature_vectors[counter] = self.create_avg_feature_vector(words)
            counter = counter + 1

        return feature_vectors

    def fit(self, raw_documents, y=None, max_features=2000, embed_dim=128):
        tokenizer = Tokenizer(num_words=max_features, split=' ')
        tokenizer.fit_on_texts(raw_documents)
        self.tokenizer = tokenizer

        X = tokenizer.texts_to_sequences(raw_documents)
        X = pad_sequences(X)

        self.model = Embedding(max_features, embed_dim, input_length=X.shape[1])

        return self

    def transform(self, raw_documents, copy=True):
        if self.model is None or self.tokenizer is None:
            print('Vectorizer is not fitted!!!')
            return

        tensors = self.tokenizer.texts_to_sequences(raw_documents)
        tensors = pad_sequences(tensors)
        X = self.model(tensors).numpy()

        return X

    def fit_transform(self, raw_documents, y=None):
        self.fit(raw_documents)

        return self.transform(raw_documents)
