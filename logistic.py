from sklearn.linear_model import LogisticRegression
from load_data import *
from scores import *
from compute_output import *


def logistic_regress(X_tr, y_tr, X_test):
    # fitting the model
    # - forcing the fit not be one vs all
    # - the default solver (lbfgs) does not converge so using a different solver
    classifier = LogisticRegression(max_iter=10000, multi_class='multinomial', solver='sag')
    classifier.fit(X_tr, y_tr)

    # finding the accuracy
    y_pred = classifier.predict(X_test)
    return y_pred


def run_logistic_once(X_tr, y_tr, X_test, y_test, extra_data, output_file):
    y_pred = logistic_regress(X_tr, y_tr, X_test)
    log_scores(y_pred, y_test, extra_data, output_file)


def run_logistic():
    compute_output_all_linear(run_logistic_once, "../results/logistic.txt")
