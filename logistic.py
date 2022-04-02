from sklearn.linear_model import LogisticRegression
from load_data import *
from scores import *


def logistic_regress(X_tr, y_tr, X_test, y_test):
    # fitting the model
    classifier = LogisticRegression(max_iter=10000, multi_class='multinomial')  # forcing the fit not be one vs all
    classifier.fit(X_tr, y_tr)

    # finding the accuracy
    y_pred = classifier.predict(X_test)
    return get_accuracy(y_pred, y_test), get_accuracy_order(y_pred, y_test), get_accuracy_one(y_pred, y_test)


def run_logistic():
    # loading the data
    X_tr, y_tr = read_training_data_linear()
    X_test, y_test = read_test_data_linear()

    sc, sc_ord, sc_one = logistic_regress(X_tr, y_tr, X_test, y_test)
    print(f"Scores are {sc}, {sc_ord}, {sc_one}")

    file = open("../results/logistic.txt", 'w')
    file.write(str(sc) + "\n")
    file.write(str(sc_ord) + "\n")
    file.write(str(sc_one) + "\n")
    file.close()
