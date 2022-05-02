from sklearn.naive_bayes import MultinomialNB
from compute_output import *
from scores import *
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


def nb_regress(X_tr, y_tr, X_test):
    # fitting the model
    classifier = MultinomialNB()
    classifier.fit(X_tr, y_tr)

    # finding the accuracy
    y_pred = classifier.predict(X_test)
    return y_pred


def run_nb_once(X_tr, y_tr, X_test, y_test, extra_data, output_file):
    y_pred = nb_regress(X_tr, y_tr, X_test)
    log_scores(y_pred, y_test, extra_data, output_file)


def run_nb():
    compute_output_all_linear(run_nb_once, "../results/nb.txt")


def save_avgs(output_dir, inds, size, X_tr, y_tr):
    y = np.asarray(y_tr)
    for i in inds:
        X = X_tr[y == i]

        X_mean = np.mean(X, axis=0)
        X_plt = np.reshape(X_mean, size)

        plt.imshow(X_plt)
        plt.title(str(i))
        plt.savefig(output_dir + str(i) + ".png")


def interpret_nb():
    # the indices to make images for
    inds = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # loading and analysing MNIST
    X_MNIST, y_MNIST = load_digits(return_X_y=True)
    save_avgs("../results/nb_anal/MNIST_", inds, (8, 8), X_MNIST, y_MNIST)

    # for getting what the size of the output image should be
    first_image_col = image.imread("../dataset/test_images/00001.png")
    first_image_gray = image.imread("../dataset/test_images_grayscale/00001.png")

    # analysing coloured images
    X_tr, y_tr = read_training_data_linear()
    save_avgs("../results/nb_anal/col_", inds, first_image_col.shape, X_tr, y_tr)

    # analysing grayscale images
    X_tr_bw, y_tr_bw = read_training_data_linear_bw()
    save_avgs("../results/nb_anal/gray_", inds, first_image_gray.shape, X_tr_bw, y_tr_bw)
