from sklearn.neighbors import KNeighborsClassifier
from scores import *
from compute_output import *


def knn(n, X_tr, y_tr, X_test):
    # fitting the model
    knn_classifier = KNeighborsClassifier(n_neighbors=n)
    knn_classifier.fit(X_tr, y_tr)

    # finding the accuracy
    y_pred = knn_classifier.predict(X_test)
    return y_pred


# a helper function for passing run_knn_once into compute_output
def helper_knn(X_tr, y_tr, X_test, y_test, extra_data, output_file, preamble):
    file = open(output_file, 'a')
    file.write("\n" + preamble + "\n")
    file.close()

    # if X_tr.shape[0] < 50000:
    #    ns = np.logspace(np.log10(5), np.log10(10000), 10)
    # else:
    #    ns = np.logspace(np.log10(500), np.log10(60000), 10)
    ns = np.logspace(np.log10(int(len(y_tr / 300))), np.log10(int(len(y_tr / 3))), 10)
    for n in ns:
        y_pred = knn(int(n), X_tr, y_tr, X_test)
        file = open(output_file, 'a')
        file.write(str(int(n)) + " & ")
        file.close()
        log_scores(y_pred, y_test, extra_data, output_file)


def run_knn():
    compute_output(helper_knn, "../results/knn.txt")
