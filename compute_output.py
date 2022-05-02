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


def compute_output_init(output_file):
    # clearing the output file
    file = open(output_file, 'w')
    file.close()


def compute_output_coloured_train_linear(func, output_file):
    # loading the coloured data
    X_tr, y_tr = read_training_data_linear()
    X_test, y_test = read_test_data_linear()
    file = open(output_file, 'a')
    file.write("\ncoloured images, training :\n")
    file.close()
    func(X_tr, y_tr, X_test, y_test, None, output_file)


def compute_output_coloured_extra_linear(func, output_file):
    # loading the coloured data
    X_tr, y_tr = read_training_data_linear()
    X_test, y_test = read_test_data_linear()
    X_extra, y_extra = read_extra_data_linear()

    # for memory efficiency defining these variables
    X_big = np.concatenate((X_tr, X_extra))
    Y_big = np.concatenate((y_tr, y_extra))
    del X_tr
    del y_tr
    del X_extra
    del y_extra

    file = open(output_file, 'a')
    file.write("\ncoloured images, training + extra : \n")
    file.close()
    func(X_big, Y_big, X_test, y_test, None, output_file)


def compute_output_grayscale_train_linear(func, output_file):
    # loading the grayscale data
    X_tr_bw, y_tr_bw = read_training_data_linear_bw()
    X_test_bw, y_test_bw = read_test_data_linear_bw()

    file = open(output_file, 'a')
    file.write("\ngrayscale images, training : \n")
    file.close()
    func(X_tr_bw, y_tr_bw, X_test_bw, y_test_bw, None, output_file)


def compute_output_grayscale_extra_linear(func, output_file):
    # loading the grayscale data
    X_tr_bw, y_tr_bw = read_training_data_linear_bw()
    X_test_bw, y_test_bw = read_test_data_linear_bw()
    X_extra_bw, y_extra_bw = read_extra_data_linear_bw()

    # for memory efficiency defining these variables
    X_big_bw = np.concatenate((X_tr_bw, X_extra_bw))
    Y_big_bw = np.concatenate((y_tr_bw, y_extra_bw))
    del X_extra_bw
    del y_extra_bw
    del X_tr_bw
    del y_tr_bw

    file = open(output_file, 'a')
    file.write("\ngrayscale images, training + extra :\n")
    file.close()
    func(X_big_bw, Y_big_bw, X_test_bw, y_test_bw, None, output_file)


def compute_output_coloured_MNIST_train_linear(func, output_file):
    # loading the MNIST style data
    X_tr_MNIST, y_tr_MNIST = read_training_data_linear_MNIST()
    X_test_MNIST, y_test_MNIST, no_digits_test = read_test_data_linear_MNIST()

    file = open(output_file, 'a')
    file.write("\ncoloured MNIST style images, training : \n")
    file.close()
    func(X_tr_MNIST, y_tr_MNIST, X_test_MNIST, y_test_MNIST, no_digits_test, output_file)


def compute_output_coloured_MNIST_extra_linear(func, output_file):
    # loading the MNIST style data
    X_tr_MNIST, y_tr_MNIST = read_training_data_linear_MNIST()
    X_test_MNIST, y_test_MNIST, no_digits_test = read_test_data_linear_MNIST()
    X_extra_MNIST, y_extra_MNIST = read_extra_data_linear_MNIST()

    # for memory efficiency defining these variables
    X_big_MNIST = np.concatenate((X_tr_MNIST, X_extra_MNIST))
    Y_big_MNIST = np.concatenate((y_tr_MNIST, y_extra_MNIST))
    del X_extra_MNIST
    del y_extra_MNIST
    del X_tr_MNIST
    del y_tr_MNIST

    file = open(output_file, 'a')
    file.write("\ncoloured MNIST style images, training + extra :\n")
    file.close()
    func(X_big_MNIST, Y_big_MNIST, X_test_MNIST, y_test_MNIST, no_digits_test, output_file)


def compute_output_grayscale_MNIST_train_linear(func, output_file):
    # loading the MNIST style data
    X_tr_MNIST, y_tr_MNIST = read_training_data_linear_MNIST_bw()
    X_test_MNIST, y_test_MNIST, no_digits_test = read_test_data_linear_MNIST_bw()

    file = open(output_file, 'a')
    file.write("\ngrayscale MNIST style images, training : \n")
    file.close()
    func(X_tr_MNIST, y_tr_MNIST, X_test_MNIST, y_test_MNIST, no_digits_test, output_file)


def compute_output_grayscale_MNIST_extra_linear(func, output_file):
    # loading the MNIST style data
    X_tr_MNIST, y_tr_MNIST = read_training_data_linear_MNIST_bw()
    X_test_MNIST, y_test_MNIST, no_digits_test = read_test_data_linear_MNIST_bw()
    X_extra_MNIST, y_extra_MNIST = read_extra_data_linear_MNIST_bw()

    # for memory efficiency defining these variables
    X_big_MNIST = np.concatenate((X_tr_MNIST, X_extra_MNIST))
    Y_big_MNIST = np.concatenate((y_tr_MNIST, y_extra_MNIST))
    del X_extra_MNIST
    del y_extra_MNIST
    del X_tr_MNIST
    del y_tr_MNIST

    file = open(output_file, 'a')
    file.write("\ngrayscale MNIST style images, training + extra :\n")
    file.close()
    func(X_big_MNIST, Y_big_MNIST, X_test_MNIST, y_test_MNIST, no_digits_test, output_file)


def compute_output_MNIST_linear(func, output_file):
    # loading the MNIST data
    X_tr_MNIST, y_tr_MNIST = read_MNIST_linear()
    X_test_MNIST, y_test_MNIST, no_digits_test = read_test_data_linear_MNIST_bw()

    file = open(output_file, 'a')
    file.write("\nMNIST :\n")
    file.close()
    func(X_tr_MNIST, y_tr_MNIST, X_test_MNIST, y_test_MNIST, no_digits_test, output_file)


def compute_output_all_linear(func, output_file):
    compute_output_init(output_file)
    compute_output_coloured_train_linear(func, output_file)
    compute_output_coloured_extra_linear(func, output_file)
    compute_output_grayscale_train_linear(func, output_file)
    compute_output_grayscale_extra_linear(func, output_file)
    compute_output_coloured_MNIST_train_linear(func, output_file)
    compute_output_coloured_MNIST_extra_linear(func, output_file)
    compute_output_grayscale_MNIST_train_linear(func, output_file)
    compute_output_grayscale_MNIST_extra_linear(func, output_file)
    compute_output_MNIST_linear(func, output_file)
