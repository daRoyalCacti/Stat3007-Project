from sklearn.ensemble import RandomForestClassifier
from scores import *
from compute_output import *
from load_data import *


def rf(X_tr, y_tr, X_test):
    # fitting the model
    rf_classifier = RandomForestClassifier(max_depth=3, random_state=42)
    rf_classifier.fit(X_tr, y_tr)

    # finding the accuracy
    y_pred = rf_classifier.predict(X_test)
    return y_pred


# a helper function for passing run_rf_once into compute_output
def run_rf_once(X_tr, y_tr, X_test, y_test, extra_data, output_file):
    y_pred = rf(X_tr, y_tr, X_test)
    log_scores(y_pred, y_test, extra_data, output_file)


def run_rf():
    compute_output_all_linear(run_rf_once, "../results/rf.txt")
