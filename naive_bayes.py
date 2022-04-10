from sklearn.naive_bayes import MultinomialNB
from compute_output import *
from scores import *


def nb_regress(X_tr, y_tr, X_test, y_test):
    # fitting the model
    classifier = MultinomialNB()
    classifier.fit(X_tr, y_tr)

    # finding the accuracy
    y_pred = classifier.predict(X_test)
    return get_accuracy(y_pred, y_test), get_accuracy_order(y_pred, y_test), get_accuracy_one(y_pred, y_test)


def run_nb_once(X_tr, y_tr, X_test, y_test, output_file, preamble):
    sc, sc_ord, sc_one = nb_regress(X_tr, y_tr, X_test, y_test)
    file = open(output_file, 'a')
    file.write("\n" + preamble + "\n")
    file.write(str(sc) + " & " + str(sc_ord) + " & " + str(sc_one) + "\n")
    file.close()


def run_nb():
    compute_output(run_nb_once, "../results/nb.txt")
