# Stat3007-project

Major project for Stat3007!

## Directory structure

````
project
│
└───Code
|
└───Results
│   
└───Dataset
    └───extra_images
    └───extra_images_grayscale
    └───extra_images_MNIST
    └───extra_images_MNIST_grayscale
    └───MNIST
    └───test_images
    └───test_images_grayscale
    .
    .
````

## How the code works

All algorithms are wrapped in a helper function, and all algorithms have their own .py file The inputs to these
functions are X_tr, y_tr, X_test, y_test, extra_data, output_file which correspond to:
the training data and labels, the test data and labels, some extra data that is need to correctly compute accuracies on
the MNIST style datasets, and an output file to write the results to.

The typical way to make a helper function is to first write a function that takes the training data and labels and the
testing data. This function will then train the model on the training data and then calculate the labels for the testing
data.

All these helper functions do is run the algorithm and pass the algorithms off to the log_scores functions to log the
results. A simple example would be

```python
def helper_function(X_tr, y_tr, X_test, y_test, extra_data, output_file):
    y_pred = predict(X_tr, y_tr, X_test)
    log_scores(y_pred, y_test, extra_data, output_file)
```

where the log_scores function is general for every algorithm and has already been written. It can be found in scores.py.
It computes the: standard accuracy, how often the prediction is just a permutation of the true result, how often one
digit was correctly identified, and how often a classifier that did not appear in the training set was classified.

There are compute_output_ functions that are used to run these functions. These functions load the data, write some
information to the output_file, and run the helper function A simple example of how to use the compute_output_ functions
is

```python
def run_algorithm():
    output_file = '../results/algorithm_name.txt'
    compute_output_init(output_file)
    compute_output_coloured_train_linear(helper_function_, output_file)
    compute_output_grayscale_train_linear(helper_function_, output_file)
    compute_output_grayscale_extra_linear(helper_function_, output_file)
```

All the compute_output_ functions are found in compute_output.py. Functions with _train_ in them only use the training
data as training data, whereas _extra_ functions use both the extra and training data as training data. Currently,
only _linear_ functions have been written. These read each image in as a long vector.








