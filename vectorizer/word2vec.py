import os
import numpy as np
from nltk.tokenize import sent_tokenize
from gensim.models.word2vec import Word2Vec

from preprocessor.preprocess_noise import PreprocessController


class Word2VecVectorizer(object):
    """
    Argument types in functions

    raw_document is a string or file iterable.
    raw_documents is a list of strings or file iterables.
    document is a list of processed words from raw_document.
    documents is a list of lists.

    """
    pc = PreprocessController()

    def __init__(self, **kwargs):
        self.model = None

    @staticmethod
    def prepare_to_fit(raw_documents):
        sentences = []
        for text in raw_documents:
            text_sentences = sent_tokenize(text)
            sentences.extend([Word2VecVectorizer.pc.process_sent(sent) for sent in text_sentences])

        return sentences

    @staticmethod
    def prepare_to_transform(raw_documents):
        cleaned_data = []

        for text in raw_documents:
            text_sentences = sent_tokenize(text)
            processed_sentences = (Word2VecVectorizer.pc.process_sent(sent, clean_up=True) for sent in text_sentences)
            prepared_text = []
            for words in processed_sentences:
                prepared_text.extend(words)

            cleaned_data.append(prepared_text)

        return cleaned_data

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

    def fit(self, raw_documents, y=None, num_features=300):
        #TODO: File path Problem
        model_file = '../vectorizer/word2vec_' + str(num_features) + '.model'
        if os.path.isfile(model_file):
            model = Word2Vec.load(model_file)
            print('Using cached ' + model_file)
        else:
            min_word_count = 40
            num_workers = 4
            window = 10
            downsampling = 1e-3

            sentences = self.prepare_to_fit(raw_documents)
            model = Word2Vec(sentences,
                             workers=num_workers,
                             size=num_features,
                             min_count=min_word_count,
                             window=window,
                             sample=downsampling)

            model.init_sims(replace=True)
            model.save(model_file)

        self.model = model
        return self

    def transform(self, raw_documents, copy=True, verbose=False):
        if self.model is None:
            print('Vectorizer is not fitted!!!')
            return

        documents = self.prepare_to_transform(raw_documents)
        X = self.get_avg_feature_vectors(documents, verbose)

        return X

    def fit_transform(self, raw_documents, y=None, verbose=False):
        self.fit(raw_documents)

        return self.transform(raw_documents, verbose)
