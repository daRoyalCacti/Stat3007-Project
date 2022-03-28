from sklearn.linear_model import LogisticRegression
from load_data import *


def logistic_regress(X_tr, y_tr, X_test, y_test):
    # fitting the model
    classifier = LogisticRegression(max_iter=10000)
    classifier.fit(X_tr, y_tr)

    # finding the accuracy
    sc = classifier.score(X_test, y_test)
    return sc


def run_logistic():
    # loading the data
    X_tr, y_tr = read_training_data_linear()
    X_test, y_test = read_test_data_linear()

    sc = logistic_regress(X_tr, y_tr, X_test, y_test)
    print(f"Score is {sc}")

    file = open("../results/logistic.txt", 'w')
    file.write(str(sc) + "\n")
    file.close()
