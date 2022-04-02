from sklearn.neighbors import KNeighborsClassifier
from load_data import *
from scores import *


def knn(n, X_tr, y_tr, X_test, y_test):
    # fitting the model
    knn_classifier = KNeighborsClassifier(n_neighbors=n)
    knn_classifier.fit(X_tr, y_tr)

    # finding the accuracy
    y_pred = knn_classifier.predict(X_test)
    return get_accuracy(y_pred, y_test), get_accuracy_order(y_pred, y_test), get_accuracy_one(y_pred, y_test)


def run_knn():
    ns = np.logspace(np.log10(5), np.log10(10000), 10)

    # loading the data
    X_tr, y_tr = read_training_data_linear()
    X_test, y_test = read_test_data_linear()

    file = open("../results/knn.txt", 'w')
    for n in ns:
        sc, sc_ord, sc_one = knn(int(n), X_tr, y_tr, X_test, y_test)
        print(f"Scores for n={int(n)} are {sc}, {sc_ord}, {sc_one}")
        file.write(str(int(n)) + ", " + str(sc) + ", " + str(sc_ord) + ", " + str(sc_one) + "\n")
    file.close()
