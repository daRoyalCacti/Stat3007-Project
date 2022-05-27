import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
from tqdm import tqdm
from compute_output import *
from load_data import *
from torch.utils.data import Dataset, DataLoader


def get_output_size(w, f, p, d, s):
    return np.floor((w + 2 * p - d * (f - 1) - 1) / s) + 1


class CNN(nn.Module):
    def __init__(self, input_shape, no_classes):
        super(CNN, self).__init__()
        self.no_class = no_classes

        filter_size1 = 5
        stride1 = 1
        padding1 = 2
        dilation1 = 1
        channels1 = 16

        filter_size2 = filter_size1
        stride2 = 1
        padding2 = 2
        dilation2 = 1
        channels2 = 32

        pooling_size = 2

        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=input_shape[2],  # input height
                out_channels=channels1,  # n_filters
                kernel_size=filter_size1,  # filter size
                stride=stride1,  # filter movement/step
                padding=padding1,
                # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
                dilation=dilation1
            ),  # output shape (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=pooling_size),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(channels1, channels2, filter_size2, stride2, padding2, dilation2),  # output shape (32, 14, 14)
            nn.ReLU(),  # activation
            nn.MaxPool2d(pooling_size),  # output shape (32, 7, 7)
        )
        # first conv layer
        output_w1 = get_output_size(input_shape[1], filter_size1, padding1, dilation1, stride1)
        output_h1 = get_output_size(input_shape[0], filter_size1, padding1, dilation1, stride1)
        # max pooling layer 1
        output_w2 = get_output_size(output_w1, pooling_size, 0, 1, pooling_size)
        output_h2 = get_output_size(output_h1, pooling_size, 0, 1, pooling_size)
        # conv layer 2
        output_w3 = get_output_size(output_w2, filter_size2, padding2, dilation2, stride2)
        output_h3 = get_output_size(output_h2, filter_size2, padding2, dilation2, stride2)
        # max pooling lyer 2
        output_w4 = get_output_size(output_w3, pooling_size, 0, 1, pooling_size)
        output_h4 = get_output_size(output_h3, pooling_size, 0, 1, pooling_size)

        self.out = nn.Linear(int(channels2 * output_w4 * output_h4),
                             int(no_classes))  # fully connected layer, output 10 classes

    def forward(self, x):
        if x.get_device() >= 0:
            output = torch.zeros(x.shape[0], self.no_class, device=x.get_device())
        else:
            output = torch.zeros(x.shape[0], self.no_class)
        for k in range(output.shape[0]):
            xp = torch.reshape(x[k], (1, x[k].shape[0], x[k].shape[1], x[k].shape[2])).float()
            xp = self.conv1(xp)
            # x = self.conv1(x)
            xp = self.conv2(xp)
            xp = xp.view(xp.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
            # output = self.out(x)
            output[k, :] = self.out(xp)
        return output


def testing():
    X_tr, y_tr = read_test_data_image()
    # X_tst, y_tst = read_test_data_image()

    CNN_test = CNN(X_tr[0].shape, len(np.unique(y_tr)))

    print(CNN_test.forward(
        torch.reshape(torch.from_numpy(X_tr[0]), (1, X_tr[0].shape[2], X_tr[0].shape[1], X_tr[0].shape[0]))))


# class for minibatching
class DatasetWrapper(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class DatasetWrapper1D(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


def learn_cnn(cnn, X_tr, y_tr, device, criterion):
    epochs = 12  # 1

    X_tr = torch.reshape(torch.from_numpy(X_tr), (X_tr.shape[0], X_tr.shape[3], X_tr.shape[2], X_tr.shape[1]))
    # y needs to be the indices in it's unique array for cross entropy to work
    y_unique = np.unique(y_tr).tolist()
    y = torch.zeros(len(y_tr))
    for i in range(len(y)):
        y[i] = y_unique.index(y_tr[i])

    # optimizer to learn the parameters of the network (I.e. the weights)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-4)
    # optimizer = torch.optim.Adam(ae.parameters(), lr=1000)

    # for mini_batching
    data_loader = DataLoader(DatasetWrapper(X_tr, y), batch_size=200, shuffle=True)

    # training loop
    for i in tqdm(range(epochs)):
        for x_batch, y_batch in data_loader:
            optimizer.zero_grad()
            # getting the results from the CNN when run on the training data
            out = cnn(x_batch.float().to(device))  # net(x_train)
            # getting the loss of this output (comparing to the labels of the training data)
            loss = criterion(out.float(), y_batch.to(int).to(device))
            # computing the gradients of the loss function
            loss.backward()
            # running the optimizer
            optimizer.step()


def learn_cnn_standard(X_tr, y_tr, X_tst, y_tst, extra_data, output_file):
    np.random.seed(0)
    torch.manual_seed(0)

    X_tst = torch.reshape(torch.from_numpy(X_tst), (X_tst.shape[0], X_tst.shape[3], X_tst.shape[2], X_tst.shape[1]))
    unique_y = np.unique(y_tr)

    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cnn = CNN(X_tr[0].shape, len(unique_y)).to(device)

    learn_cnn(cnn, X_tr, y_tr, device, torch.nn.CrossEntropyLoss())

    res = torch.zeros(len(y_tst))
    # get training set accuracy
    data_loader = DataLoader(DatasetWrapper1D(X_tst), batch_size=200, shuffle=True)
    counter = 0
    for x_batch in data_loader:
        gpu_x = x_batch.float().to(device)
        # getting the results from the CNN when run on the training data
        # need to take the most likely one
        output = cnn(gpu_x).detach().cpu()
        for i in range(output.shape[0]):
            res[counter] = unique_y[np.argmax(output[i]).item()]  # net(x_train)
            counter += 1

    log_scores(res.detach().numpy().astype(int), y_tst, extra_data, output_file)


def run_cnn_standard():
    compute_output_all_image(learn_cnn_standard, "../results/cnn.txt")
