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
def helper_knn(X_tr, y_tr, X_test, y_test, extra_data, output_file):
    # ns = np.logspace(np.log10(int(len(y_tr) / 3000)), np.log10(int(len(y_tr) / 300)), 4)
    ns = [1, 10, 20, 50, 100]
    for n in ns:
        y_pred = knn(int(n), X_tr, y_tr, X_test)
        file = open(output_file, 'a')
        file.write(str(int(n)) + " & ")
        file.close()
        log_scores(y_pred, y_test, extra_data, output_file)


def run_knn():
    compute_output_all_linear(helper_knn, "../results/knn.txt")
