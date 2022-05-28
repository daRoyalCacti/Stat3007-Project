import torch
from torch.utils.data import Dataset, DataLoader
from load_data import *
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from compute_output import *


class sparse_autoencoder(torch.nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.encoder_layers = torch.nn.ModuleList()
        self.decoder_layers = torch.nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.encoder_layers.append(torch.nn.Linear(int(layer_sizes[i]), int(layer_sizes[i + 1]), bias=True))
            self.decoder_layers.append(torch.nn.Linear(int(layer_sizes[-i - 1]), int(layer_sizes[-i - 2]), bias=True))
        self.activation = torch.nn.LeakyReLU()
        self.encoder_size = int(layer_sizes[-1])

    def forward(self, x):
        if x.get_device() >= 0:
            output = torch.zeros(x.shape, device=x.get_device())
            latent = torch.zeros(x.shape[0], self.encoder_size, device=x.get_device())
        else:
            output = torch.zeros(x.shape)
            latent = torch.zeros(x.shape[0], self.encoder_size)
        for j in range(output.shape[0]):
            temp = x[j]
            for i in range(len(self.encoder_layers)):
                temp = self.activation(self.encoder_layers[i](temp))
            latent[j] = temp
            for i in range(len(self.decoder_layers)):
                temp = self.activation(self.decoder_layers[i](temp))
            output[j] = temp
        return latent, output

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


def KL_divergence(p, q, batch_size):
    """
    Calculate the KL-divergence of (p,q)
    :param p:
    :param q:
    :return:
    """
    q = torch.nn.functional.softmax(q, dim=0)
    q = torch.sum(q, dim=0) / batch_size
    s1 = torch.sum(p * torch.log(p / q))
    s2 = torch.sum((1 - p) * torch.log((1 - p) / (1 - q)))
    return s1 + s2


# class for minibatching
class DatasetWrapperAE(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


def learn_ae_kl(ae, X_tr, X_tst, device, criterion):
    epochs = 100
    bs = 10
    expect_tho = 0.05
    beta = 10

    tho_tensor = torch.FloatTensor([expect_tho for _ in range(int(ae.encoder_size))])
    # if torch.cuda.is_available():
    #    tho_tensor = tho_tensor.cuda()

    # optimizer to learn the parameters of the network (I.e. the weights)
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-4)
    # optimizer = torch.optim.Adam(ae.parameters(), lr=1000)

    # for mini_batching
    data_loader = DataLoader(DatasetWrapperAE(X_tr), batch_size=bs, shuffle=True)

    # training loop
    for i in tqdm(range(epochs)):
        for x_batch in data_loader:
            optimizer.zero_grad()
            gpu_x = x_batch.float().to(device)
            # getting the results from the CNN when run on the training data
            latent, out = ae(gpu_x)  # net(x_train)
            # getting the loss of this output (comparing to the labels of the training data)
            loss = criterion(out, gpu_x)
            # adding the extra kl term to the loss
            kl = KL_divergence(tho_tensor, latent, bs)
            loss += beta * kl
            # computing the gradients of the loss function
            loss.backward()
            # running the optimizer
            optimizer.step()

        # saving the latent vectors for all images after every epoch
        file = open("../results/ae_anal/standard_kl/latent_vectors_train_epoch_" + str(i) + ".txt", 'w')
        for k in range(X_tr.shape[0]):
            input_v = torch.reshape(torch.from_numpy(X_tr[k]), (1, X_tr[k].shape[0])).float()
            latent_v = ae.get_latent_vector(input_v).detach().cpu().numpy()
            latent_v = latent_v[0]
            for j in latent_v:
                file.write(str(j) + " ")
            file.write("\n")
        file.close()
        file = open("../results/ae_anal/standard_kl/latent_vectors_test_epoch_" + str(i) + ".txt", 'w')
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
            _, X_comp = ae(torch.reshape(torch.from_numpy(X_tr[k]), (1, X_tr[k].shape[0])).float())
            X_comp = np.reshape(X_comp.detach().cpu().numpy(), (30, 60, 3))
            file = open("../results/ae_anal/standard_kl/image" + str(k) + "_epoch" + str(i) + ".txt", 'w')
            file.write(str(X_comp))
            file.close()

            plt.imshow(X_comp)
            # hiding the axes
            ax = plt.gca()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            plt.savefig("../results/ae_anal/standard_kl/image" + str(k) + "_epoch" + str(i) + ".png",
                        bbox_inches='tight',
                        pad_inches=0)


def learn_ae_kl_standard():
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

    ae = sparse_autoencoder(layers).float().to(device)

    learn_ae_kl(ae, X_tr, X_tst, device, torch.nn.MSELoss())

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

        plt.savefig("../results/ae_anal/standard_kl/images/" + str(ind) + "_uncompressed.png", bbox_inches='tight',
                    pad_inches=0)

        # saving the compressed plot
        X_comp = ae(torch.reshape(torch.from_numpy(X_tr[ind]), (1, X_tr[ind].shape[0])).float())
        X_comp = np.reshape(X_comp.detach().cpu().numpy(), (30, 60, 3))
        plt.imshow(X_comp)
        # hiding the axes
        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.savefig("../results/ae_anal/standard_kl/images/" + str(ind) + "_compressed.png", bbox_inches='tight',
                    pad_inches=0)
