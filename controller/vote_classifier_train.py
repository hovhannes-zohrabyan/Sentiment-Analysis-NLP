import time
from controller.data_controller import DataRW
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB


class VoteClassifierTrain:
    """
      Model is class intended to read data and train loaded and preprocessed data and saving them with picklke
      This class also trains clarification models and saves them
      """

    def __init__(self):
        print("Training started at " + str(time.strftime("%H:%M:%S", time.localtime())))
        self.data_controller = DataRW()

    def train_models(self, x_train=None, y_train=None):

        # models = [('bernoulli_nb_classifier', BernoulliNB()), ('logistic_regression_classifier', LogisticRegression()),
        #           ('linear_svc_classifier', LinearSVC(random_state=0, tol=1e-5)),
        #           ('sgd_classifier', SGDClassifier(max_iter=1000, tol=1e-3))]
        models = [('linear_svc_classifier', LinearSVC(random_state=0, tol=1e-5))]

        for name, model in models:
            model.fit(x_train, y_train)
            self.data_controller.save_data_pickle('trained_model', name, model)

    def main_controller(self, x_train, y_train):
        self.train_models(x_train, y_train)
        print("Training ended at " + str(time.strftime("%H:%M:%S", time.localtime())))


# if __name__ == '__main__':
#     train = ModelTrain()
    # train.main_controller()
