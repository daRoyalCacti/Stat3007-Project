import numpy as np
from matplotlib import image
import torch


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
        file_name = img_loc + str(i + 1).zfill(5) + ".png"
        img = image.imread(file_name)
        X[i, :] = np.reshape(img, (1, first_image.size))

    return X, y


def read_data_image(img_loc, label):
    # reading the identifiers
    y = read_identifies(label)

    # reading the image data
    first_image = image.imread(img_loc + "00001.png")

    if len(first_image.shape) == 3:
        X = np.zeros((len(y), first_image.shape[0], first_image.shape[1], first_image.shape[2]))
    elif len(first_image.shape) == 2:
        X = np.zeros((len(y), first_image.shape[0], first_image.shape[1], 1))
    else:
        print("IMAGE IS NOT AN IMAGE")
        X = []

    for i in range(1, len(y)):
        file_name = img_loc + str(i + 1).zfill(5) + ".png"
        img = image.imread(file_name)
        if len(first_image.shape) == 3:
            X[i, :, :, :] = img
        else:
            X[i, :, :, :] = np.reshape(img, (img.shape[0], img.shape[1], 1))

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
    return read_data_linear("../dataset/training_images_MNIST/", "../dataset/training_labels_MNIST.txt")


def read_extra_data_linear_MNIST():
    return read_data_linear("../dataset/extra_images_MNIST/", "../dataset/extra_labels_MNIST.txt")


def read_test_data_linear_MNIST_bw():
    X, y = read_data_linear("../dataset/test_images_MNIST_grayscale/", "../dataset/test_labels_MNIST.txt")
    return X, y, read_identifies("../dataset/test_no_digits.txt")


def read_training_data_linear_MNIST_bw():
    return read_data_linear("../dataset/training_images_MNIST_grayscale/", "../dataset/training_labels_MNIST.txt")


def read_extra_data_linear_MNIST_bw():
    return read_data_linear("../dataset/extra_images_MNIST_grayscale/", "../dataset/extra_labels_MNIST.txt")


def read_MNIST_linear():
    return read_data_linear("../dataset/MNIST/", "../dataset/MNIST_labels.txt")


def read_latent_vectors(file_loc):
    # read in the data
    with open(file_loc, 'r') as g:
        str_lines = g.readlines()
    y = torch.zeros((len(str_lines), 200))
    for i in range(len(str_lines)):
        arr_str = str_lines[i]
        # get the indices of all the spaces
        spaces = []
        for pos, char in enumerate(arr_str):
            if char == ' ':
                spaces.append(pos)
        y[i, 0] = float(arr_str[0:spaces[0]])
        for j in range(len(spaces) - 1):
            y[i, j + 1] = float(arr_str[spaces[j]:spaces[j + 1]])
    return y


def read_test_data_image():
    return read_data_image("../dataset/test_images/", "../dataset/test_labels.txt")


def read_training_data_image():
    return read_data_image("../dataset/training_images/", "../dataset/training_labels.txt")


def read_extra_data_image():
    return read_data_image("../dataset/extra_images/", "../dataset/extra_labels.txt")


# functions for reading the grayscale images in as vectors
def read_test_data_image_bw():
    return read_data_image("../dataset/test_images_grayscale/", "../dataset/test_labels.txt")


def read_training_data_image_bw():
    return read_data_image("../dataset/training_images_grayscale/", "../dataset/training_labels.txt")


def read_extra_data_image_bw():
    return read_data_image("../dataset/extra_images_grayscale/", "../dataset/extra_labels.txt")


# functions for reading the MNIST style images in as vectors
def read_test_data_image_MNIST():
    X, y = read_data_image("../dataset/test_images_MNIST/", "../dataset/test_labels_MNIST.txt")
    return X, y, read_identifies("../dataset/test_no_digits.txt")


def read_training_data_image_MNIST():
    return read_data_image("../dataset/training_images_MNIST/", "../dataset/training_labels_MNIST.txt")


def read_extra_data_image_MNIST():
    return read_data_image("../dataset/extra_images_MNIST/", "../dataset/extra_labels_MNIST.txt")


def read_test_data_image_MNIST_bw():
    X, y = read_data_image("../dataset/test_images_MNIST_grayscale/", "../dataset/test_labels_MNIST.txt")
    return X, y, read_identifies("../dataset/test_no_digits.txt")


def read_training_data_image_MNIST_bw():
    return read_data_image("../dataset/training_images_MNIST_grayscale/", "../dataset/training_labels_MNIST.txt")


def read_extra_data_image_MNIST_bw():
    return read_data_image("../dataset/extra_images_MNIST_grayscale/", "../dataset/extra_labels_MNIST.txt")


def read_MNIST_image():
    return read_data_image("../dataset/MNIST/", "../dataset/MNIST_labels.txt")
