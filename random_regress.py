from load_data import *
import numpy as np


def random_regress(y_test):
    sc = 0

    for i in range(len(y_test)):
        if np.random.randint(0, 10) == y_test[i]:
            sc += 1

    return sc / len(y_test)


def run_random_regress():
    # loading the data
    X_test, y_test = read_test_data_linear()

    sc = random_regress(y_test)
    print(f"Score is {sc}")

    file = open("../results/random.txt", 'w')
    file.write(str(sc) + "\n")
    file.close()
