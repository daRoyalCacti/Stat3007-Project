import numpy as np
from matplotlib import image


def read_identifies(label):
    with open(label, 'r') as g:
        y = list(map(int, g.readlines()))
    return y


def read_data_linear(img_loc, label):
    # reading the identifiers
    y = read_identifies(label)

    # reading the image data and flattening to a 1D array
    first_image = image.imread(img_loc + "00001.png")
    first_image = np.reshape(first_image, (1, first_image.size))

    X = np.zeros((len(y), first_image.size))
    X[0, :] = first_image

    for i in range(1, len(y)):
        file_name = img_loc + str(i).zfill(5) + ".png"
        img = image.imread(file_name)
        X[i, :] = np.reshape(img, (1, first_image.size))

    return X, y


# functions for reading the coloured images in as vectors
def read_test_data_linear():
    return read_data_linear("../dataset/test_images/", "../dataset/test_labels.txt")


def read_training_data_linear():
    return read_data_linear("../dataset/training_images/", "../dataset/training_labels.txt")


def read_extra_data_linear():
    return read_data_linear("../dataset/extra_images/", "../dataset/extra_labels.txt")


# functions for reading the grayscale images in as vectors
def read_test_data_linear_bw():
    return read_data_linear("../dataset/test_images_grayscale/", "../dataset/test_labels.txt")


def read_training_data_linear_bw():
    return read_data_linear("../dataset/training_images_grayscale/", "../dataset/training_labels.txt")


def read_extra_data_linear_bw():
    return read_data_linear("../dataset/extra_images_grayscale/", "../dataset/extra_labels.txt")


# functions for reading the MNIST style images in as vectors
def read_test_data_linear_MNIST():
    X, y = read_data_linear("../dataset/test_images_MNIST/", "../dataset/test_labels_MNIST.txt")
    return X, y, read_identifies("../dataset/test_no_digits.txt")


def read_training_data_linear_MNIST():
    X, y = read_data_linear("../dataset/training_images_MNIST/", "../dataset/training_labels_MNIST.txt")
    return X, y, read_identifies("../dataset/training_no_digits.txt")


def read_extra_data_linear_MNIST():
    X, y = read_data_linear("../dataset/extra_images_MNIST/", "../dataset/extra_labels_MNIST.txt")
    return X, y, read_identifies("../dataset/extra_no_digits.txt")
