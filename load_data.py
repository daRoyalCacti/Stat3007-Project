import numpy as np
from matplotlib import image


def read_data_linear(img_loc, label):
    # reading the identifiers
    with open(label, 'r') as g:
        y = list(map(int, g.readlines()))

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


def read_test_data_linear():
    return read_data_linear("../dataset/test_images/", "../dataset/test_labels.txt")


def read_training_data_linear():
    return read_data_linear("../dataset/training_images/", "../dataset/training_labels.txt")


def read_extra_data_linear():
    return read_data_linear("../dataset/extra_images/", "../dataset/extra_labels.txt")
