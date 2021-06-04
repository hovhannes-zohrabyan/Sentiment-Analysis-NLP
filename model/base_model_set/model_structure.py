from controller.data_controller import DataRW


class BaseModelStructure:

    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.data_controller = DataRW()

    def fit(self, x_train, y_train):
        # TODO: The way date is given has to be changed
        train_data = list(map(lambda x, y: (x, y), x_train, y_train))
        classifier = self.model.train(train_data)
        classifier.train(train_data)
        self.data_controller.save_data_pickle("trained_model", self.model_name, classifier)
        return classifier

    def predict(self, features):
        try:
            base_classifier = self.data_controller.read_data_pickle("trained_model", "base_classifier")
            result = base_classifier.classify(features)
            return result
        except FileNotFoundError:
            return "Model Not Found"

    def return_model(self):
        try:
            base_classifier = self.data_controller.read_data_pickle("trained_model", "base_classifier")
            return base_classifier
        except FileNotFoundError:
            return "Model Not Found"
