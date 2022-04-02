from sklearn.neighbors import KNeighborsClassifier
from load_data import *


def knn(n, X_tr, y_tr, X_test, y_test):
    # fitting the model
    knn_classifier = KNeighborsClassifier(n_neighbors=n)
    knn_classifier.fit(X_tr, y_tr)

    # finding the accuracy
    sc = knn_classifier.score(X_test, y_test)
    return sc


def run_knn():
    ns = np.logspace(np.log10(5), np.log10(10000), 10)

    # loading the data
    X_tr, y_tr = read_training_data_linear()
    X_test, y_test = read_test_data_linear()

    scs = np.zeros(len(ns))

    i = 0
    for n in ns:
        sc = knn(int(n), X_tr, y_tr, X_test, y_test)
        print(f"Score for n={int(n)} is {sc}")
        scs[i] = sc
        i += 1

    file = open("../results/knn.txt", 'w')
    for i in range(len(ns)):
        file.write(str(int(ns[i])) + ", " + str(scs[i]) + "\n")
    file.close()
