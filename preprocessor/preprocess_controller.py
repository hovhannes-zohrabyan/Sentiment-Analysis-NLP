import nltk
from nltk import word_tokenize
from controller.data_controller import DataRW
from sklearn import model_selection


class PreprocessController:

    def __init__(self):
        self.data_controller = DataRW()
        self.documents = []

    def find_features(self, document):
        words = word_tokenize(document)
        features = {}
        for word in self.word_features:
            sad = word in words
            features[word] = sad

        return features

    def prepare_data(self, dataset):
        vocab_words = []
        gen_data = dataset

        # allowed_word_types = ["J", "R", "V"]

        for data_type in gen_data:
            for p in data_type[0][:10]:
                self.documents.append((p, data_type[1]))
                words = word_tokenize(p)
                pos = nltk.pos_tag(words)
                for w in pos:
                    # if w[1][0] in allowed_word_types:
                    vocab_words.append(w[0].lower())

        allowed_word_types = ["J", "R", "V"]

        # Save Data
        self.data_controller.save_data_pickle("data", "documents", self.documents)

        # Get Most Frequently used words
        vocab_words = nltk.FreqDist(vocab_words)

        self.word_features = list(vocab_words.keys())[:5000]

        self.data_controller.save_data_pickle("data", "word_features", self.word_features)

        feature_sets = {
                "feature": [],
                "category": []
            }

        for rev, category in self.documents:
            # for i in range(len(rev)):
            #     features = self.find_features(rev[i])
            feature_sets["feature"].append(rev)
            feature_sets["category"].append(category)

        x_train, x_test, y_train, y_test = model_selection.train_test_split(feature_sets["feature"], feature_sets["category"], test_size=0.2)

        return x_train, x_test, y_train, y_test
