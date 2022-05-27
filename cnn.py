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

        self.last_layer_size = int(channels2 * output_w4 * output_h4)

        self.out = nn.Linear(self.last_layer_size, int(no_classes))  # fully connected layer, output 10 classes

    def get_last_layer(self, x):
        if x.get_device() >= 0:
            output = torch.zeros(x.shape[0], self.last_layer_size, device=x.get_device())
        else:
            output = torch.zeros(x.shape[0], self.last_layer_size)
        for k in range(output.shape[0]):
            xp = torch.reshape(x[k], (1, x[k].shape[0], x[k].shape[1], x[k].shape[2])).float()
            xp = self.conv1(xp)
            # x = self.conv1(x)
            xp = self.conv2(xp)
            xp = xp.view(xp.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
            # output = self.out(x)
            output[k, :] = xp
        return output

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
    interpret_cnn(cnn, X_tr, device)


global_counter = 0


def interpret_cnn(cnn, X_tr, device):
    X_tr = torch.reshape(torch.from_numpy(X_tr), (X_tr.shape[0], X_tr.shape[3], X_tr.shape[2], X_tr.shape[1]))
    res = torch.zeros(X_tr.shape[0], cnn.last_layer_size)
    data_loader = DataLoader(DatasetWrapper1D(X_tr), batch_size=200, shuffle=True)
    counter = 0
    for x_batch in data_loader:
        gpu_x = x_batch.float().to(device)
        # getting the results from the CNN when run on the training data
        # need to take the most likely one
        output = cnn.get_last_layer(gpu_x).detach().cpu()
        for i in range(output.shape[0]):
            res[counter] = output[i]  # net(x_train)
            counter += 1
    res = res.detach().numpy()
    print_weights(cnn)
    get_nearest_neighbours(res, [i for i in range(20)], 50)


def print_weights(cnn):
    global global_counter
    file = open('../results/cnn_anal/standard/' + str(global_counter) + '_weights.txt', 'w')

    w1 = cnn.conv1[0].weight.data
    w2 = cnn.conv2[0].weight.data

    file.write("w1s\n===============================================\n")
    for i in range(w1.shape[0]):
        for j in range(w1.shape[1]):
            file.write("Matrix : (" + str(i) + "," + str(j) + ")\n")
            for k in range(w1.shape[2]):
                for h in range(w1.shape[3]):
                    file.write(str(w1[i, j, k, h].item()) + ", ")
                file.write("\n")
    file.write("w2s\n===============================================\n")
    for i in range(w2.shape[0]):
        for j in range(w2.shape[1]):
            file.write("Matrix : (" + str(i) + "," + str(j) + ")\n")
            for k in range(w2.shape[2]):
                for h in range(w2.shape[3]):
                    file.write(str(w2[i, j, k, h].item()) + ", ")
                file.write("\n")

    file.close()


def get_nearest_neighbours(res, inds, k):
    global global_counter
    errors = np.zeros((len(inds), res.shape[0]))
    for i in range(len(inds)):
        img = res[inds[i]]
        for j in range(res.shape[0]):
            if i == j:
                errors[i, j] = 1000000000
            else:
                errors[i, j] = np.sum((img - res[j]) ** 2)

    file = open('../results/cnn_anal/standard/' + str(global_counter) + '.txt', 'w')
    for i in range(len(inds)):
        file.write("Nearest neighbours for image : " + str(inds[i]) + "\n")
        closest_inds = np.argsort(errors[i])
        for j in range(k):
            file.write(str(j) + " closest : " + str(closest_inds[j]) + "\n")
        file.write("\n")
    file.close()
    global_counter += 1


def run_cnn_standard():
    global global_counter
    global_counter = 0
    compute_output_all_image(learn_cnn_standard, "../results/cnn.txt")
    global_counter = 0
