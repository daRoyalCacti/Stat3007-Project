from sklearn.neural_network import MLPClassifier
from scores import *
from compute_output import *
from load_data import *


def MLP(X_tr, y_tr, X_test):
    # fitting the model
    MLP_classifier = MLPClassifier(batch_size=128, activation='relu',max_iter=10000,random_state=42,learning_rate='adaptive',early_stopping=True)
    MLP_classifier.fit(X_tr, y_tr)

    # finding the accuracy
    y_pred = MLP_classifier.predict(X_test)
    return y_pred


# a helper function for passing run_MLP_once into compute_output
def run_MLP_once(X_tr, y_tr, X_test, y_test, extra_data, output_file):
    y_pred = MLP(X_tr, y_tr, X_test)
    log_scores(y_pred, y_test, extra_data, output_file)


def run_MLP():
    compute_output_all_linear(run_MLP_once, "../results/MLP.txt")
