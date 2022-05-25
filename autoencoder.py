import torch
from torch.utils.data import Dataset, DataLoader
from load_data import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from compute_output import *


class autoencoder(torch.nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.encoder_layers = torch.nn.ModuleList()
        self.decoder_layers = torch.nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.encoder_layers.append(torch.nn.Linear(int(layer_sizes[i]), int(layer_sizes[i + 1]), bias=True))
            self.decoder_layers.append(torch.nn.Linear(int(layer_sizes[-i - 1]), int(layer_sizes[-i - 2]), bias=True))
        self.activation = torch.nn.LeakyReLU()

    def forward(self, x):
        if x.get_device() >= 0:
            output = torch.zeros(x.shape, device=x.get_device())
        else:
            output = torch.zeros(x.shape)
        for j in range(output.shape[0]):
            temp = x[j]
            for i in range(len(self.encoder_layers)):
                temp = self.activation(self.encoder_layers[i](temp))
            for i in range(len(self.decoder_layers)):
                temp = self.activation(self.decoder_layers[i](temp))
            output[j] = temp
        return output

    def get_latent_vector(self, x):
        if x.get_device() >= 0:
            output = torch.zeros((x.shape[0], self.encoder_layers[-1].out_features), device=x.get_device())
        else:
            output = torch.zeros((x.shape[0], self.encoder_layers[-1].out_features))
        for j in range(output.shape[0]):
            temp = x[j]
            for i in range(len(self.encoder_layers)):
                temp = self.activation(self.encoder_layers[i](temp))
            output[j] = temp
        return output


class MLP_AE(torch.nn.Module):
    def __init__(self, latent_vec_size):
        super().__init__()
        # self.linear1 = torch.nn.Linear(latent_vec_size, latent_vec_size)
        self.linear2 = torch.nn.Linear(latent_vec_size, latent_vec_size // 2)
        self.linear3 = torch.nn.Linear(latent_vec_size // 2, 55)
        self.activation = torch.nn.LeakyReLU()
        self.softmax = torch.nn.Softmax(0)

    def forward(self, x):
        # output_shape = (x.shape[0], 1000000 - 1)
        output_size1 = x.shape[0]
        output_size2 = 100000 - 1
        # output_shape = x.shape[0]
        if x.get_device() >= 0:
            output = torch.zeros(output_size1, output_size2, device=x.get_device())
        else:
            output = torch.zeros(output_size1, output_size2)
        for k in range(output.shape[0]):
            # x1 = self.activation(self.linear1(x[k]))
            # x2 = self.activation(self.linear2(x1))
            x2 = self.activation(self.linear2(x[k]))
            x3 = self.linear3(x2)

            x3[0:5] = self.softmax(x3[0:5])
            for i in range(5):
                x3[(10 * i + 5):(10 * i + 15)] = self.softmax(x3[(10 * i + 5):(10 * i + 15)])

            for i in range(output_size2):
                # get the number of digits
                if i == 0:
                    n = 1
                else:
                    n = int(np.log10(i))

                # get each digit out
                '''digits = np.zeros(n + 1)
                digits[0] = int(i / 10 ** n)
                for j in range(n):
                    to_clear = int(i / 10 ** (n - j)) * 10 ** (n - j)
                    digits[j + 1] = int((i - to_clear) / 10 ** (n - j - 1))'''
                digits = [int(x) for x in str(i)]
                # multiply the probabilities together
                output[k, i] = x3[n]
                for j in range(n):
                    output[k, i] *= x3[5 + int(digits[j]) + 10 * j]
        return output

    def run_test(self, x):
        output = np.zeros(x.shape[0])
        with torch.no_grad():
            for k in range(output.shape[0]):
                # x1 = self.activation(self.linear1(x[k]))
                # x2 = self.activation(self.linear2(x1))
                x2 = self.activation(self.linear2(x[k]))
                x3 = self.linear3(x2)

                # predicting the digit from the last layer
                no_digits = np.argmax(x3[0:5].numpy()) + 1

                pred = 0
                for i in range(no_digits):
                    # getting the prediction for each digit
                    max_ind = 0
                    for j in range(10):
                        if x3[10 * i + 5 + j] > x3[10 * i + 5 + max_ind]:
                            max_ind = j
                    # appending this digit to the current prediction
                    pred = 10 * pred + max_ind
                output[k] = pred
                # res = self.forward( torch.reshape(x[k], (1, len(x[k]))) )
                # output[k] = np.argmax(res.detach().numpy())
        # for k in range(3):
        #    print(output[k])
        return output.astype(int)


# class for minibatching
class DatasetWrapperAE(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


class DatasetWrapperMLP(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def learn_ae(ae, X_tr, X_tst, device, criterion):
    epochs = 100

    # optimizer to learn the parameters of the network (I.e. the weights)
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-4)
    # optimizer = torch.optim.Adam(ae.parameters(), lr=1000)

    # for mini_batching
    data_loader = DataLoader(DatasetWrapperAE(X_tr), batch_size=10, shuffle=True)

    # training loop
    for i in tqdm(range(epochs)):
        for x_batch in data_loader:
            optimizer.zero_grad()
            gpu_x = x_batch.float().to(device)
            # getting the results from the CNN when run on the training data
            out = ae(gpu_x)  # net(x_train)
            # getting the loss of this output (comparing to the labels of the training data)
            loss = criterion(out, gpu_x)
            # computing the gradients of the loss function
            loss.backward()
            # running the optimizer
            optimizer.step()
        # saving the latent vectors for all images after every epoch
        file = open("../results/ae_anal/standard/latent_vectors_train_epoch_" + str(i) + ".txt", 'w')
        for k in range(X_tr.shape[0]):
            input_v = torch.reshape(torch.from_numpy(X_tr[k]), (1, X_tr[k].shape[0])).float()
            latent_v = ae.get_latent_vector(input_v).detach().cpu().numpy()
            latent_v = latent_v[0]
            for j in latent_v:
                file.write(str(j) + " ")
            file.write("\n")
        file.close()
        file = open("../results/ae_anal/standard/latent_vectors_test_epoch_" + str(i) + ".txt", 'w')
        for k in range(X_tst.shape[0]):
            input_v = torch.reshape(torch.from_numpy(X_tst[k]), (1, X_tst[k].shape[0])).float()
            latent_v = ae.get_latent_vector(input_v).detach().cpu().numpy()
            latent_v = latent_v[0]
            for j in latent_v:
                file.write(str(j) + " ")
            file.write("\n")
        file.close()

        draw_inds = [0, 1, 3, 5, 7, 8, 92]
        for k in draw_inds:
            # saving the compressed plot for some images
            X_comp = ae(torch.reshape(torch.from_numpy(X_tr[k]), (1, X_tr[k].shape[0])).float())
            X_comp = np.reshape(X_comp.detach().cpu().numpy(), (30, 60, 3))
            file = open("../results/ae_anal/standard/image" + str(k) + "_epoch" + str(i) + ".txt", 'w')
            file.write(str(X_comp))
            file.close()

            plt.imshow(X_comp)
            # hiding the axes
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            plt.savefig("../results/ae_anal/standard/image" + str(k) + "_epoch" + str(i) + ".png", bbox_inches='tight',
                        pad_inches=0)


def learn_ae_standard():
    np.random.seed(0)
    torch.manual_seed(0)

    X_tr, y_tr = read_training_data_linear()
    X_tst, y_tst = read_test_data_linear()
    del y_tr
    del y_tst

    layers = np.linspace(200, X_tr.shape[1], 3)
    layers = layers[::-1]

    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae = autoencoder(layers).float().to(device)

    learn_ae(ae, X_tr, X_tst, device, torch.nn.MSELoss())

    # saving the images of some results
    np.random.seed(0)
    for i in range(300):
        ind = np.random.randint(0, X_tr.shape[0])

        # saving the uncompressed plot
        X_plt = np.reshape(X_tr[ind], (30, 60, 3))
        plt.imshow(X_plt)
        # hiding the axes
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.savefig("../results/ae_anal/standard/images/" + str(ind) + "_uncompressed.png", bbox_inches='tight',
                    pad_inches=0)

        # saving the compressed plot
        X_comp = ae(torch.reshape(torch.from_numpy(X_tr[ind]), (1, X_tr[ind].shape[0])).float())
        X_comp = np.reshape(X_comp.detach().cpu().numpy(), (30, 60, 3))
        plt.imshow(X_comp)
        # hiding the axes
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.savefig("../results/ae_anal/standard/images/" + str(ind) + "_compressed.png", bbox_inches='tight',
                    pad_inches=0)


def classify_ae_standard():
    X_tr = read_latent_vectors("../results/ae_anal/standard/latent_vectors_train_epoch_19.txt")
    X_ts = read_latent_vectors("../results/ae_anal/standard/latent_vectors_test_epoch_19.txt")
    _, y_tr = read_training_data_linear()
    _, y_tst = read_test_data_linear()
    del _

    # X_tr = X_tr[0:3]
    # X_ts = X_ts[0:3]
    # y_tr = y_tr[0:3]
    # y_tst = y_tst[0:3]

    output_file = "../results/ae_standard.txt"

    np.random.seed(0)
    torch.manual_seed(0)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    MLP = MLP_AE(200).float().to(device)

    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.CrossEntropyLoss()

    epochs = 10000

    # optimizer to learn the parameters of the network (I.e. the weights)
    # optimizer = torch.optim.Adam(MLP.parameters(), lr=1e-4)
    # optimizer = torch.optim.Adam(MLP.parameters(), lr=1000)
    optimizer = torch.optim.Adadelta(MLP.parameters())

    # for mini_batching
    data_loader = DataLoader(DatasetWrapperMLP(X_tr, y_tr), batch_size=1, shuffle=True)

    # clearing the output file
    file = open(output_file, 'w')
    file.close()

    # training loop
    for i in tqdm(range(epochs)):
        # run the optimizer
        counter = 0
        for x_batch, y_batch in data_loader:
            if y_batch > 100000 - 2:
                continue
            counter += 1
            optimizer.zero_grad()
            # getting the results from the CNN when run on the training data
            out = MLP(x_batch.float().to(device))  # net(x_train)
            # getting the loss of this output (comparing to the labels of the training data)
            loss = criterion(out, y_batch)
            # computing the gradients of the loss function
            loss.backward()
            # running the optimizer
            optimizer.step()

            # get the predicted results on the test set
            # doing it for every batch just to get some output
            # MLP_CPU = MLP.cpu()
            # y_pred = MLP_CPU.run_test(X_ts)
            y_pred = MLP.run_test(X_ts)
            # save the results
            file = open(output_file, 'a')
            file.write("Epoch " + str(i + 1) + ", Batch " + str(counter) + " : ")
            file.close()
            log_scores(y_pred, y_tst, None, output_file)


def learn_ae_l2(ae, X_tr, X_tst, device, criterion):
    epochs = 100

    # optimizer to learn the parameters of the network (I.e. the weights)
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-4, weight_decay=0.00001)
    # optimizer = torch.optim.Adam(ae.parameters(), lr=1000)

    # for mini_batching
    data_loader = DataLoader(DatasetWrapperAE(X_tr), batch_size=10, shuffle=True)

    # training loop
    for i in tqdm(range(epochs)):
        for x_batch in data_loader:
            optimizer.zero_grad()
            gpu_x = x_batch.float().to(device)
            # getting the results from the CNN when run on the training data
            out = ae(gpu_x)  # net(x_train)
            # getting the loss of this output (comparing to the labels of the training data)
            loss = criterion(out, gpu_x)
            # computing the gradients of the loss function
            loss.backward()
            # running the optimizer
            optimizer.step()
        # saving the latent vectors for all images after every epoch
        file = open("../results/ae_anal/standard_l2/latent_vectors_train_epoch_" + str(i) + ".txt", 'w')
        for k in range(X_tr.shape[0]):
            input_v = torch.reshape(torch.from_numpy(X_tr[k]), (1, X_tr[k].shape[0])).float()
            latent_v = ae.get_latent_vector(input_v).detach().cpu().numpy()
            latent_v = latent_v[0]
            for j in latent_v:
                file.write(str(j) + " ")
            file.write("\n")
        file.close()
        file = open("../results/ae_anal/standard_l2/latent_vectors_test_epoch_" + str(i) + ".txt", 'w')
        for k in range(X_tst.shape[0]):
            input_v = torch.reshape(torch.from_numpy(X_tst[k]), (1, X_tst[k].shape[0])).float()
            latent_v = ae.get_latent_vector(input_v).detach().cpu().numpy()
            latent_v = latent_v[0]
            for j in latent_v:
                file.write(str(j) + " ")
            file.write("\n")
        file.close()

        draw_inds = [0, 1, 3, 5, 7, 8, 92]
        for k in draw_inds:
            # saving the compressed plot for some images
            X_comp = ae(torch.reshape(torch.from_numpy(X_tr[k]), (1, X_tr[k].shape[0])).float())
            X_comp = np.reshape(X_comp.detach().cpu().numpy(), (30, 60, 3))
            file = open("../results/ae_anal/standard_l2/image" + str(k) + "_epoch" + str(i) + ".txt", 'w')
            file.write(str(X_comp))
            file.close()

            plt.imshow(X_comp)
            # hiding the axes
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            plt.savefig("../results/ae_anal/standard_l2/image" + str(k) + "_epoch" + str(i) + ".png",
                        bbox_inches='tight',
                        pad_inches=0)


def learn_ae_l2_standard():
    np.random.seed(0)
    torch.manual_seed(0)

    X_tr, y_tr = read_training_data_linear()
    X_tst, y_tst = read_test_data_linear()
    del y_tr
    del y_tst

    layers = np.linspace(200, X_tr.shape[1], 3)
    layers = layers[::-1]

    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ae = autoencoder(layers).float().to(device)

    learn_ae_l2(ae, X_tr, X_tst, device, torch.nn.MSELoss())

    # saving the images of some results
    np.random.seed(0)
    for i in range(300):
        ind = np.random.randint(0, X_tr.shape[0])

        # saving the uncompressed plot
        X_plt = np.reshape(X_tr[ind], (30, 60, 3))
        plt.imshow(X_plt)
        # hiding the axes
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.savefig("../results/ae_anal/standard_l2/images/" + str(ind) + "_uncompressed.png", bbox_inches='tight',
                    pad_inches=0)

        # saving the compressed plot
        X_comp = ae(torch.reshape(torch.from_numpy(X_tr[ind]), (1, X_tr[ind].shape[0])).float())
        X_comp = np.reshape(X_comp.detach().cpu().numpy(), (30, 60, 3))
        plt.imshow(X_comp)
        # hiding the axes
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.savefig("../results/ae_anal/standard_l2/images/" + str(ind) + "_compressed.png", bbox_inches='tight',
                    pad_inches=0)
