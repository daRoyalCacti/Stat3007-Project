import sys

from knn import *
from naive_bayes import *
from random_regress import *
from logistic import *
from autoencoder import *
from randomforest import *
from MLP import *
from cnn import *
from sparse_autoencoder import *
from sparse_autoencoder2 import *
from deeper_cnn import *
from plot_cnn_weights import *

from data_analysis import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # get_untrainable_digits()
    run_MLP()
    run_knn()
    # run_random_regress()
    run_nb()
    run_rf()
    # run_logistic()
    # interpret_nb()

    # learn_ae_standard()
    # classify_ae_standard()
    # learn_ae_l2_standard()
    # learn_ae_kl_standard()
    # learn_ae_kl_standard2()
    # run_MLP()
    # run_cnn_deeper()

    # plot_cnn_weights('../results/cnn_anal/standard/0_weights.txt', '../results/cnn_anal/standard/weight_images.png', 6, 6)
    # plot_cnn_weights('../results/cnn_anal/deeper/105_weights.txt', '../results/cnn_anal/deeper/weight_images.png', 3, 3)

    # get_nearest_neighbours('../dataset/training_images_MNIST/', '../results/cnn_anal/deeper/105.txt', '../results/cnn_anal/deeper/MNIST_style/')

    # run_cnn_standard()

    print("done")
    sys.exit()
