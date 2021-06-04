import pickle
import os


class DataRW:

    # TODO: Upgrade paths to abs
    @staticmethod
    def save_data_pickle(directory, name, model):
        abs_directory = os.path.join('model_data', directory, name + '.pickle')
        doc = open(abs_directory, "wb")
        pickle.dump(model, doc)
        doc.close()

    @staticmethod
    def read_data_pickle(directory, name):
        abs_directory = os.path.join('model_data', directory, name + '.pickle')
        if os.path.isfile(abs_directory):
            pickle_file = open(abs_directory, "rb")
            classifier = pickle.load(pickle_file)
            pickle_file.close()
        else:
            raise FileNotFoundError

        return classifier
