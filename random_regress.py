from load_data import *
from scores import *
from compute_output import *


def random_regress(y_test, digits):
    preds = np.zeros(len(y_test))
    for i in range(len(y_test)):
        preds[i] = digits[np.random.randint(0, len(digits))]

    return preds


# a helper function for passing run_knn_once into compute_output
def helper_random_regress(X_tr, y_tr, X_test, y_test, extra_data, output_file):
    if extra_data is None:
        if X_tr.shape[0] < 50000:
            with open("../results/unique_digits_train.txt", 'r') as g:
                digits = list(map(int, g.readlines()))
        else:
            with open("../results/unique_digits.txt", 'r') as g:
                digits = list(map(int, g.readlines()))
    else:
        digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    y_pred = random_regress(y_test, digits)
    log_scores(y_pred, y_test, extra_data, output_file)


def run_random_regress():
    compute_output_all_linear(helper_random_regress, "../results/random.txt")
