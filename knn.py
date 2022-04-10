from sklearn.neighbors import KNeighborsClassifier
from scores import *
from compute_output import *


def knn(n, X_tr, y_tr, X_test, y_test):
    # fitting the model
    knn_classifier = KNeighborsClassifier(n_neighbors=n)
    knn_classifier.fit(X_tr, y_tr)

    # finding the accuracy
    y_pred = knn_classifier.predict(X_test)
    return get_accuracy(y_pred, y_test), get_accuracy_order(y_pred, y_test), get_accuracy_one(y_pred, y_test)


def run_knn_once(ns, X_tr, y_tr, X_test, y_test, output_file, preamble):
    file = open(output_file, 'a')
    file.write("\n" + preamble + "\n")
    for n in ns:
        sc, sc_ord, sc_one = knn(int(n), X_tr, y_tr, X_test, y_test)
        file.write(str(int(n)) + ", " + str(sc) + ", " + str(sc_ord) + ", " + str(sc_one) + "\n")
        file.flush()
    file.close()


# a helper function for passing run_knn_once into compute_output
def helper_knn(X_tr, y_tr, X_test, y_test, output_file, preamble):
    if X_tr.size > 50000:
        return run_knn_once(np.logspace(np.log10(5), np.log10(10000), 10), X_tr, y_tr, X_test, y_test, output_file,
                            preamble)
    else:
        return run_knn_once(np.logspace(np.log10(500), np.log10(100000), 10), X_tr, y_tr, X_test, y_test, output_file,
                            preamble)


def run_knn():
    compute_output(helper_knn, "../results/nb.txt")
