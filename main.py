import sys

from knn import *
from naive_bayes import *
from random_regress import *
from logistic import *
from autoencoder import *

from data_analysis import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # get_untrainable_digits()
    # run_random_regress()
    # run_nb()
    # run_knn()
    # run_logistic()
    # interpret_nb()

    # test = autoencoder([2, 30, 15])
    # test(torch.zeros((1,1,2)))
    interpret_ae_standard()

    print("done")
    sys.exit()
