from load_data import *


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

    func(X_tr, y_tr, X_test, y_test, output_file, "coloured images, training :")
    # training data is no longer needed
    del X_tr
    del y_tr

    func(X_big, Y_big, X_test, y_test, output_file, "coloured images, training + extra : ")

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

    func(X_tr_bw, y_tr_bw, X_test_bw, y_test_bw, output_file, "grayscale images, training : ")
    # training data is no longer needed
    del X_tr_bw
    del y_tr_bw

    func(X_big_bw, Y_big_bw, X_test_bw, y_test_bw, output_file, "grayscale images, training + extra :")
