import csv
import os
import pandas as pd


# TODO: Change to one united dataset
class DatasetController:

    def __init__(self):
        self.absolute_path = os.path.abspath(os.path.join('../data', 'main_dataset', 'all.csv'))

    def import_dataset(self):

        pos_data, neg_data, neutral_data = self.import_general_train_data()

        general_data = [(pos_data, "pos"), (neg_data, "neg")]

        return general_data

    def import_general_train_data(self):
        pos_data = []
        neg_data = []
        neutral_data = []

        with open(self.absolute_path, "r", encoding="utf-8") as main_dataset:
            prep_pos_data = csv.reader(main_dataset, delimiter=',')
            for row in prep_pos_data:
                # print(row[2], row[0])
                if row[2] == "1":
                    pos_data.append(row[0])
                elif row[2] == "0":
                    neg_data.append(row[0])
                elif row[2] == "2":
                    neutral_data.append(row[0])
            # print(pos_data[:1], neg_data[:1], neutral_data[:1], sep="\n")

        return pos_data, neg_data, neutral_data

    @staticmethod
    def get_data(filename):
        train_data = pd.read_csv(os.path.join('data', 'main_dataset', filename))

        return train_data['Text'].values, train_data['Label'].values
