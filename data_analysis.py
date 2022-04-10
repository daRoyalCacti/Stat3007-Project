import collections
import numpy as np


def get_number_frequencies(input_data, output_data, min_digit):
    with open(input_data, 'r') as g:
        y = list(map(int, g.readlines()))
    freq = collections.Counter(y)

    numbers = []
    freqs = []

    for (key, value) in freq.items():
        numbers.append(key)
        freqs.append(value)

    inds = np.argsort(-np.asarray(freqs))

    file = open(output_data, 'w')
    j = 0
    for i in range(20):
        while numbers[inds[j]] < min_digit:
            j += 1
        file.write(str(numbers[inds[j]]) + "," + str(freqs[inds[j]]) + "\n")
        j += 1
    file.close()


def get_number_frequencies_1_digit(input_data, output_data):
    get_number_frequencies(input_data, output_data, 0)


def get_number_frequencies_3_digit(input_data, output_data):
    get_number_frequencies(input_data, output_data, 100)


def get_number_frequencies_4_digit(input_data, output_data):
    get_number_frequencies(input_data, output_data, 1000)


def get_number_frequencies_all():
    get_number_frequencies_1_digit("../dataset/test_labels.txt", "../results/freqs/testing.txt")
    get_number_frequencies_3_digit("../dataset/test_labels.txt", "../results/freqs/testing_3.txt")
    get_number_frequencies_4_digit("../dataset/test_labels.txt", "../results/freqs/testing_4.txt")

    get_number_frequencies_1_digit("../dataset/training_labels.txt", "../results/freqs/training.txt")
    get_number_frequencies_3_digit("../dataset/training_labels.txt", "../results/freqs/training_3.txt")
    get_number_frequencies_4_digit("../dataset/training_labels.txt", "../results/freqs/training_4.txt")

    get_number_frequencies_1_digit("../dataset/extra_labels.txt", "../results/freqs/extra.txt")
    get_number_frequencies_3_digit("../dataset/extra_labels.txt", "../results/freqs/extra_3.txt")
    get_number_frequencies_4_digit("../dataset/extra_labels.txt", "../results/freqs/extra_4.txt")


def get_unique_numbers():
    with open("../dataset/test_labels.txt", 'r') as g:
        y1 = list(map(int, g.readlines()))
    with open("../dataset/training_labels.txt", 'r') as g:
        y2 = list(map(int, g.readlines()))
    with open("../dataset/extra_labels.txt", 'r') as g:
        y3 = list(map(int, g.readlines()))
    y = y1 + y2 + y2
    unique_y = np.unique(np.array(y))

    file = open("../results/unique_digits.txt", 'w')
    for u in unique_y:
        file.write(str(u) + "\n")
    file.close()


def get_unique_numbers_train():
    with open("../dataset/training_labels.txt", 'r') as g:
        y = list(map(int, g.readlines()))

    unique_y = np.unique(np.array(y))
    file = open("../results/unique_digits_train.txt", 'w')
    for u in unique_y:
        file.write(str(u) + "\n")
    file.close()
