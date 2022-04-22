from load_data import *
from scores import *


def log_scores(y_pred, y_test, extra_data, output_file):
    sc = get_accuracy(y_pred, y_test, extra_data)
    sc_ord = get_accuracy_order(y_pred, y_test, extra_data)
    sc_one = get_accuracy_one(y_pred, y_test, extra_data)
    sc_unt = get_accuracy_untrainable(y_pred, y_test, extra_data)

    file = open(output_file, 'a')
    file.write(str(sc) + " & " + str(sc_ord) + " & " + str(sc_one) + " & " + str(sc_unt) + "\\\\ \n")
    file.close()


def compute_output(func, output_file):
    # clearing the output file
    file = open(output_file, 'w')
    file.close()

    # loading the coloured data
    X_tr, y_tr = read_training_data_linear()
    X_test, y_test = read_test_data_linear()
    X_extra, y_extra = read_extra_data_linear()

    # for memory efficiency defining these variables
    X_big = np.concatenate((X_tr, X_extra))
    Y_big = np.concatenate((y_tr, y_extra))

    del X_extra
    del y_extra

    func(X_tr, y_tr, X_test, y_test, None, output_file, "coloured images, training :")
    # training data is no longer needed
    del X_tr
    del y_tr

    func(X_big, Y_big, X_test, y_test, None, output_file, "coloured images, training + extra : ")

    # deleting the coloured data to save on ram
    del X_test
    del y_test
    del X_big
    del Y_big

    # loading the grayscale data
    X_tr_bw, y_tr_bw = read_training_data_linear_bw()
    X_test_bw, y_test_bw = read_test_data_linear_bw()
    X_extra_bw, y_extra_bw = read_extra_data_linear_bw()

    # for memory efficiency defining these variables
    X_big_bw = np.concatenate((X_tr_bw, X_extra_bw))
    Y_big_bw = np.concatenate((y_tr_bw, y_extra_bw))

    del X_extra_bw
    del y_extra_bw

    func(X_tr_bw, y_tr_bw, X_test_bw, y_test_bw, None, output_file, "grayscale images, training : ")
    # training data is no longer needed
    del X_tr_bw
    del y_tr_bw

    func(X_big_bw, Y_big_bw, X_test_bw, y_test_bw, None, output_file, "grayscale images, training + extra :")

    del X_test_bw
    del y_test_bw
    del X_big_bw
    del Y_big_bw

    # loading the MNIST style data
    X_tr_MNIST, y_tr_MNIST, no_digits_train = read_training_data_linear_MNIST()
    X_test_MNIST, y_test_MNIST, no_digits_test = read_test_data_linear_MNIST()
    X_extra_MNIST, y_extra_MNIST, no_digits_extra = read_extra_data_linear_MNIST()

    del no_digits_train
    del no_digits_extra

    # for memory efficiency defining these variables
    X_big_MNIST = np.concatenate((X_tr_MNIST, X_extra_MNIST))
    Y_big_MNIST = np.concatenate((y_tr_MNIST, y_extra_MNIST))

    del X_extra_MNIST
    del y_extra_MNIST

    func(X_tr_MNIST, y_tr_MNIST, X_test_MNIST, y_test_MNIST, no_digits_test, output_file,
         "MNIST style images, training : ")
    # training data is no longer needed
    del X_tr_MNIST
    del y_tr_MNIST

    func(X_big_MNIST, Y_big_MNIST, X_test_MNIST, y_test_MNIST, no_digits_test, output_file,
         "MNIST style images, training + extra :")

    del X_test_MNIST
    del y_test_MNIST
    del X_big_MNIST
    del Y_big_MNIST
    del no_digits_test
