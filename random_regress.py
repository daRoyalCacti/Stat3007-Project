from load_data import *
from scores import *
from compute_output import *


def random_regress(y_test, digits):
    sc = 0

    preds = np.zeros(len(y_test))
    for i in range(len(y_test)):
        preds[i] = digits[np.random.randint(0, len(digits))]

    return get_accuracy(preds, y_test), get_accuracy_order(preds, y_test), get_accuracy_one(preds,
                                                                                            y_test), get_accuracy_untrainable(
        preds, y_test)


def run_random_regress_once(digits, y_test, output_file, preamble):
    sc, sc_ord, sc_one, sc_unt = random_regress(y_test, digits)
    file = open(output_file, 'a')
    file.write("\n" + preamble + "\n")
    file.write(str(sc) + " & " + str(sc_ord) + " & " + str(sc_one) + " & " + str(sc_unt) + "\\\\ \n")
    file.close()


# a helper function for passing run_knn_once into compute_output
def helper_random_regress(X_tr, y_tr, X_test, y_test, output_file, preamble):
    if X_tr.shape[0] < 50000:
        with open("../results/unique_digits_train.txt", 'r') as g:
            digits = list(map(int, g.readlines()))
    else:
        with open("../results/unique_digits.txt", 'r') as g:
            digits = list(map(int, g.readlines()))
    return run_random_regress_once(digits, y_test, output_file, preamble)


def run_random_regress():
    compute_output(helper_random_regress, "../results/random.txt")


def run_random_regress2():
    # loading the data
    X_test, y_test = read_test_data_linear()

    # load the possible digits only in the training set
    with open("../results/unique_digits_train.txt", 'r') as g:
        digits = list(map(int, g.readlines()))

    sc, sc_ord, sc_one = random_regress(y_test, digits)
    print(f"Scores are {sc}, {sc_ord}, {sc_one}")

    file = open("../results/random.txt", 'w')
    file.write(str(sc) + "\n")
    file.write(str(sc_ord) + "\n")
    file.write(str(sc_one) + "\n")

    # load all the possible digits
    with open("../results/unique_digits.txt", 'r') as g:
        digits = list(map(int, g.readlines()))

    sc, sc_ord, sc_one = random_regress(y_test, digits)
    print(f"Scores are {sc}, {sc_ord}, {sc_one}")

    file.write("\n")
    file.write(str(sc) + "\n")
    file.write(str(sc_ord) + "\n")
    file.write(str(sc_one) + "\n")

    file.close()
