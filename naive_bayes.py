from sklearn.naive_bayes import MultinomialNB
from load_data import *


def nb_regress(X_tr, y_tr, X_test, y_test):
    # fitting the model
    classifier = MultinomialNB()
    classifier.fit(X_tr, y_tr)

    # finding the accuracy
    sc = classifier.score(X_test, y_test)
    return sc


def run_nb():
    # loading the data
    X_tr, y_tr = read_training_data_linear()
    X_test, y_test = read_test_data_linear()

    sc = nb_regress(X_tr, y_tr, X_test, y_test)
    print(f"Score is {sc}")

    file = open("../results/nb.txt", 'w')
    file.write(str(sc) + "\n")
    file.close()
