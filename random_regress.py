from load_data import *
import numpy as np


def random_regress(y_test, digits):
    sc = 0

    for i in range(len(y_test)):
        if digits[np.random.randint(0, len(digits))] == y_test[i]:
            sc += 1

    return sc / len(y_test)


def run_random_regress():
    # loading the data
    X_test, y_test = read_test_data_linear()

    # load the possible digits
    with open("../results/unique_digits.txt", 'r') as g:
        digits = list(map(int, g.readlines()))

    sc = random_regress(y_test, digits)
    print(f"Score is {sc}")

    file = open("../results/random.txt", 'w')
    file.write(str(sc) + "\n")
    file.close()
