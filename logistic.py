from sklearn.linear_model import LogisticRegression
from load_data import *
from scores import *


def logistic_regress(X_tr, y_tr, X_test, y_test):
    # fitting the model
    # - forcing the fit not be one vs all
    # - the default solver (lbfgs) does not converge so using a different solver
    classifier = LogisticRegression(max_iter=10000, multi_class='multinomial', solver='sag')
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

    # load the extra data
    X_extra, y_extra = read_extra_data_linear()

    # running the regressor on the extra data
    sc, sc_ord, sc_one = logistic_regress(np.concatenate((X_tr, X_extra)), np.concatenate((y_tr, y_extra)), X_test,
                                          y_test)
    print(f"Scores are {sc}, {sc_ord}, {sc_one}")

    file = open("../results/logistic.txt", 'a')
    file.write("\n")
    file.write(str(sc) + "\n")
    file.write(str(sc_ord) + "\n")
    file.write(str(sc_one) + "\n")
    file.close()
