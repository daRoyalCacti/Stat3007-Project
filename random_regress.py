from load_data import *
import numpy as np
from scores import *


def random_regress(y_test, digits):
    sc = 0

    preds = np.zeros(len(y_test))
    for i in range(len(y_test)):
        preds[i] = digits[np.random.randint(0, len(digits))]

    return get_accuracy(preds, y_test), get_accuracy_order(preds, y_test), get_accuracy_one(preds, y_test)


def run_random_regress():
    # loading the data
    X_test, y_test = read_test_data_linear()

    # load the possible digits
    with open("../results/unique_digits.txt", 'r') as g:
        digits = list(map(int, g.readlines()))

    sc, sc_ord, sc_one = random_regress(y_test, digits)
    print(f"Scores are {sc}, {sc_ord}, {sc_one}")

    file = open("../results/random.txt", 'w')
    file.write(str(sc) + "\n")
    file.write(str(sc_ord) + "\n")
    file.write(str(sc_one) + "\n")
    file.close()
