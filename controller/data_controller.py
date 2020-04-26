from nltk import FreqDist
import pickle


class DataRW:
    #TODO: Upgrade paths to abs
    # def __init__(self):
    #     print("Hello")

    @staticmethod
    def read_dataset(dataset, labeled=False):
        if labeled:
            pos_data = open("../datasets/" + dataset + "/positive.txt", "r").read()
            neg_data = open("../datasets/" + dataset + "/negative.txt", "r").read()
            return pos_data, neg_data

    @staticmethod
    def save_data_pickle(directory, name, model):
        doc = open("../model_data/" + directory + "/" + name + ".pickle", "wb")
        pickle.dump(model, doc)
        doc.close()

    @staticmethod
    def read_data_pickle(directory, name):
        pickle_file = open("model_data/" + directory + "/" + name + ".pickle", "rb")
        classifier = pickle.load(pickle_file)
        pickle_file.close()

        return classifier

