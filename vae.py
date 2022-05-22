import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200

import os
import tqdm
import math
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class VariationalAutoencoder(nn.Module):
    def __init__(self, image_size, num_channel, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = Encoder(image_size, num_channel, latent_dims)
        self.decoder = Decoder(image_size, num_channel, latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class Encoder(nn.Module):
    def __init__(self, image_size, num_channel, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(pow(image_size,2) * num_channel, 512)
        self.linear2 = nn.Linear(512, latent_dims) # for mean
        self.linear3 = nn.Linear(512, latent_dims) # for variance

        self.N = torch.distributions.Normal(0, 1) # zero mean, unit variance gaussian dist
        if device == 'cuda':
            self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()

        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))

        mu =  self.linear2(x) # mean
        sigma = torch.exp(self.linear3(x)) # variance
        z = mu + sigma * self.N.sample(mu.shape) # sampled with reparm trick 
        
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum() # closed-form kl divergence solution
        
        return z


class Decoder(nn.Module):
    def __init__(self, image_size, num_channel, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, pow(image_size,2) * num_channel)
        self.image_size = image_size
        self.num_channel = num_channel

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, self.num_channel, self.image_size, self.image_size))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, default=25)
    parser.add_argument('--latent_dims', type=int, default=2)
    parser.add_argument('--dataset_type', type=str, default='mnist', choices = ['mnist','cifar10'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--plot', action='store_true')

    args = parser.parse_args()


    if args.dataset_type == "mnist":
        dataset = torchvision.datasets.MNIST('./data',
                   train=True,
                   transform=torchvision.transforms.ToTensor(),
                   download=True)
        dataset_test = torchvision.datasets.MNIST('./data',
                   train=False,
                   transform=torchvision.transforms.ToTensor(),
                   download=True)
        image_size = 28
        num_channel = 1

    elif args.dataset_type == "cifar10":
        dataset = torchvision.datasets.CIFAR10('./data',
                train=True,
                transform=torchvision.transforms.ToTensor(),
                download=True)
        dataset_test = torchvision.datasets.CIFAR10('./data',
                train=False,
                transform=torchvision.transforms.ToTensor(),
                download=True)
        image_size = 32
        num_channel = 3

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,num_workers=4)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,num_workers=4)

    model = VariationalAutoencoder(image_size, num_channel, args.latent_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    criterion = nn.MSELoss(reduction='sum')

    for epoch in range(args.num_epoch):
        num_batches = len(data_loader)
        
        model.train()
        for i, (x,y) in enumerate(data_loader):
            lr = lr_scheduler.get_last_lr()[0]

            x = x.to(device)
            pred = model(x)
            loss = criterion(pred, x) + model.encoder.kl
            # import pdb; pdb.set_trace()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % args.log_interval == 0 and i > 0:
                print(f'| epoch {epoch:3d} | ' 
                    f'{i:5d}/{num_batches:5d} batches | '
                    f'lr {lr:02.4f} | '
                    f'loss {loss:5.3f}')
        
        model.eval()
        val_loss = 0.0
        for i, (x,y) in enumerate(data_loader_test):
            x = x.to(device)
            with torch.no_grad(): 
                pred = model(x)
                loss = criterion(pred, x) + model.encoder.kl
                val_loss += loss.item()
        print(f'| epoch {epoch:3d} | total validation loss {val_loss:5.2f}')
        
        lr_scheduler.step()

    if args.plot:
        save_dir_path = os.path.join(os.getcwd(),'assets')
        if not os.path.isdir(save_dir_path) : os.mkdir(save_dir_path)
        plot_latent(model, data_loader_test, save_dir_path)
        plot_reconstructed(model, image_size, num_channel, save_dir_path)


def plot_latent(model, data_loader, save_dir_path, num_batches=128):
    plt.clf()
    for i, (x, y) in enumerate(data_loader):
        z = model.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
    plt.colorbar()
    file_path = os.path.join(save_dir_path,'latent_space.png')
    if os.path.exists(file_path) : os.remove(file_path)
    plt.savefig(file_path)


def plot_reconstructed(model, image_size, num_channel, save_dir_path, r0=(-5, 10), r1=(-10, 5), n=12):
    plt.clf()
    w = image_size
    img = np.zeros((n*w, n*w, num_channel))
    for i, y in enumerate(np.linspace(*r1, n)):
        for j, x in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[x, y]]).to(device)
            x_hat = model.decoder(z)
            x_hat = x_hat.reshape(num_channel, image_size, image_size)
            recon_img = x_hat.permute(1, 2, 0).to('cpu').detach().numpy()
            img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w, :] = recon_img
    plt.imshow(img, extent=[*r0, *r1])
    file_path = os.path.join(save_dir_path,'reconstruct_images.png')
    if os.path.exists(file_path) : os.remove(file_path)
    plt.savefig(file_path)


if __name__ == "__main__":
    main()