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

from data_analysis import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # get_untrainable_digits()
    run_MLP()
    run_knn()
    run_random_regress()
    run_nb()
    run_rf()
    # run_logistic()
    # interpret_nb()

    # learn_ae_standard()
    # classify_ae_standard()
    # learn_ae_l2_standard()
    # learn_ae_kl_standard()
    # run_MLP()

    # run_cnn_standard()

    print("done")
    sys.exit()
