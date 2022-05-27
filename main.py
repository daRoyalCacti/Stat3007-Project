import sys

from knn import *
from naive_bayes import *
from random_regress import *
from logistic import *
from autoencoder import *
from randomforest import *
from MLP import *
from cnn import *

from data_analysis import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # get_untrainable_digits()
    # run_random_regress()
    # run_nb()
    # run_knn()
    # run_logistic()
    # run_rf()
    # run_MLP()
    # interpret_nb()

    # learn_ae_standard()
    # classify_ae_standard()
    # learn_ae_l2_standard()
    # run_MLP()

    run_cnn_standard()

    print("done")
    sys.exit()
