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
from PIL import Image
from sklearn import manifold, datasets
import torchvision.transforms as transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class VariationalAutoencoder(nn.Module):
    def __init__(self, image_size, num_channel, latent_dims, use_deep_layers):
        super(VariationalAutoencoder, self).__init__()
        if not use_deep_layers : 
            self.encoder = Encoder(image_size, num_channel, latent_dims) 
            self.decoder = Decoder(image_size, num_channel, latent_dims)
        else:
            self.encoder = DeepEncoder(image_size, num_channel, latent_dims) 
            self.decoder = DeepDecoder(image_size, num_channel, latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class Encoder(nn.Module):   
    def __init__(self, image_size, num_channel, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(image_size**2 * num_channel, 512)
        self.linear_mean = nn.Linear(512, latent_dims) # for mean
        self.linear_variance = nn.Linear(512, latent_dims) # for variance

        self.N = torch.distributions.Normal(0, 1) # zero mean, unit variance gaussian dist
        if device == 'cuda':
            self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()

        self.kld = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))

        mu = self.linear_mean(x) # mean
        sigma = torch.exp(self.linear_variance(x)) # variance
        z = mu + sigma * self.N.sample(mu.shape) # sampled with reparm trick 
        
        self.kld = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum() # closed-form kl divergence solution
        
        return z


class Decoder(nn.Module):
    def __init__(self, image_size, num_channel, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, image_size**2 * num_channel)
        self.image_size = image_size
        self.num_channel = num_channel

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z)) if self.num_channel == 1 else torch.sigmoid(self.linear2(z))
        return z.reshape((-1, self.num_channel, self.image_size, self.image_size))


class DeepEncoder(nn.Module):   
    def __init__(self, image_size, num_channel, latent_dims):
        super(DeepEncoder, self).__init__()

        modules = []
        hidden_dims = [32, 64, 128, 256, 512] if image_size == 64 else [64, 128, 256, 512]

        in_channel = num_channel
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channel = h_dim

        self.encoder = nn.Sequential(*modules)
        self.linear_mean = nn.Linear(hidden_dims[-1]*4, latent_dims)
        self.linear_variance = nn.Linear(hidden_dims[-1]*4, latent_dims)

        self.N = torch.distributions.Normal(0, 1) # zero mean, unit variance gaussian dist

        if device == 'cuda':
            self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()

        self.kld = 0

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)

        mu = self.linear_mean(x) # mean
        sigma = torch.exp(self.linear_variance(x)) # variance
        z = mu + sigma * self.N.sample(mu.shape) # sampled with reparm trick 
        
        self.kld = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum() # closed-form kl divergence solution
        
        return z


class DeepDecoder(nn.Module):
    def __init__(self, image_size, num_channel, latent_dims):
        super(DeepDecoder, self).__init__()

        self.image_size = image_size
        self.num_channel = num_channel

        modules = []
        hidden_dims = [512, 256, 128, 64, 32] if image_size == 64 else [512, 256, 128, 64]

        self.decoder_input = nn.Linear(latent_dims, hidden_dims[0] * 4)

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*modules)

        # 28, 32 image size
        final_conv_layer =  nn.Conv2d(
            hidden_dims[-1], 
            out_channels=num_channel,
            kernel_size=5 if image_size == 28 else 3 , 
            padding=0 if image_size == 28 else 1
            )
        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            final_conv_layer
                            )

    def forward(self, z):
        out = self.decoder_input(z)
        out = out.view(-1, 512, 2, 2)
        out = self.decoder(out)
        out = self.final_layer(out)
        out = torch.sigmoid(out) if self.num_channel == 1 else torch.sigmoid(out)
        return out

# vae_mnist_shallow_model_latent_dim128_latent_space_epoch25
# vae_mnist_shallow_model_latent_dim128_reconstruct_images_epoch25
# vae_mnist_shallow_model_latent_dim128_reconstruct_from_images_epoch25
# vae_mnist_shallow_model_latent_dim128_sample_from_prior_epoch25

# vae_cifar10_shallow_model_latent_dim128_reconstruct_from_images_epoch100
# vae_cifar10_shallow_model_latent_dim128_sample_from_prior_epoch100

# vae_cifar10_deep_model_latent_dim128_reconstruct_from_images_epoch100
# vae_cifar10_deep_model_latent_dim128_sample_from_prior_epoch100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir_path', type=str, default='./data')
    parser.add_argument('--dataset_name', type=str, default='mnist', choices = ['mnist', 'cifar10', 'celeba'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epoch', type=int, default=100)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_step', type=float, default=1.0)
    parser.add_argument('--latent_dims', type=int, default=128)
    parser.add_argument('--kld_weight', type=float, default=0.00025)
    parser.add_argument('--use_deep_layers', action='store_true')

    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--plot', action='store_true')

    args = parser.parse_args()


    if args.dataset_name == "mnist":
        image_size = 28
        num_channel = 1
        dataset = torchvision.datasets.MNIST(args.dataset_dir_path,
                   train=True,
                   transform=torchvision.transforms.ToTensor(),
                   download=True)
        dataset_test = torchvision.datasets.MNIST(args.dataset_dir_path,
                   train=False,
                   transform=torchvision.transforms.ToTensor(),
                   download=True)
    elif args.dataset_name == "cifar10":
        image_size = 32
        num_channel = 3
        dataset = torchvision.datasets.CIFAR10(args.dataset_dir_path,
                train=True,
                transform=torchvision.transforms.ToTensor(),
                download=True)
        dataset_test = torchvision.datasets.CIFAR10(args.dataset_dir_path,
                train=False,
                transform=torchvision.transforms.ToTensor(),
                download=True)
    elif args.dataset_name == "celeba":
        dataset_path = os.path.join(args.dataset_dir_path, 'celeba')
        assert os.path.isdir(dataset_path), 'You should download CelebA dataset manually in {}.'.format(args.dataset_dir_path)

        image_size = 64
        num_channel = 3
        import torchvision.transforms as transforms
        dataset = torchvision.datasets.ImageFolder(root=dataset_path,
                                transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))

        dataset_test = torchvision.datasets.ImageFolder(root=dataset_path,
                                transform=transforms.Compose([
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))

        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:-int(len(indices)*0.05)])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-int(len(indices)*0.05):])

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True,num_workers=4)
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,num_workers=4)

    model = VariationalAutoencoder(image_size, num_channel, args.latent_dims, args.use_deep_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(args.num_epoch*args.lr_step), gamma=0.333)

    for epoch in range(args.num_epoch):
        train_num_batches = len(data_loader)
        test_num_batches = len(data_loader_test)
        
        model.train()
        lr = lr_scheduler.get_last_lr()[0]
        for i, (x,y) in enumerate(data_loader):

            x = x.to(device)
            pred = model(x)
            recon_loss = loss = F.mse_loss(pred, x, reduction='sum')
            kld_loss = model.encoder.kld
            loss = recon_loss + args.kld_weight * kld_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % args.log_interval == 0 and i > 0:
                print(f'| epoch {epoch:3d} | ' 
                    f'{i:5d}/{train_num_batches:5d} batches | '
                    f'lr {lr:02.4f} | '
                    f'loss {loss:5.3f} | '
                    f'recon loss {recon_loss:5.3f} | '
                    f'kld loss {kld_loss:5.3f}')
        
        model.eval()
        val_recon_loss = 0.0
        val_kld_loss = 0.0
        val_loss = 0.0
        for i, (x,y) in enumerate(data_loader_test):
            x = x.to(device)
            with torch.no_grad(): 
                pred = model(x)
                recon_loss = F.mse_loss(pred, x, reduction='sum')
                kld_loss = model.encoder.kld
                loss = recon_loss + args.kld_weight * kld_loss

                val_recon_loss += recon_loss.item()
                val_kld_loss += kld_loss.item()
                val_loss += loss.item()

        print(f'| epoch {epoch:3d} | '
            f'lr {lr:02.4f} | '
            f'total valid loss {val_loss/test_num_batches:5.3f} | '
            f'total valid recon loss {val_recon_loss/test_num_batches:5.3f} | '
            f'total valid kld loss {val_kld_loss/test_num_batches:5.3f}')
        
        lr_scheduler.step()

    if args.plot:
        save_dir_path = os.path.join(os.getcwd(),'assets')
        if not os.path.isdir(save_dir_path) : os.mkdir(save_dir_path)

        # only bivariate prior can visualize latent vectors in 2-dim
        if args.latent_dims == 2:
            plot_latent(model, data_loader_test, save_dir_path)
            plot_reconstructed(model, image_size, num_channel, save_dir_path)

        plot_latent_with_tsne(model, data_loader_test, save_dir_path, dim=2)
        if args.latent_dims > 2:
            plot_latent_with_tsne(model, data_loader_test, save_dir_path, dim=3)

        plot_random_sample_from_prior(model, args.latent_dims, save_dir_path)
        plot_reconstruct_from_images(model, data_loader_test, save_dir_path)

        x, y = data_loader_test.__iter__().next() # hack to grab a batch
        x1 = x[y == 1][1].unsqueeze(0).to(device) # find a 1
        x2 = x[y == 0][1].unsqueeze(0).to(device) # find a 0

        interpolate(model, image_size, num_channel, save_dir_path, x1, x2)
        interpolate_gif(model, image_size, num_channel, save_dir_path, x1, x2)


def plot_latent(model, data_loader, save_dir_path):
    plt.clf()
    for i, (x, y) in enumerate(data_loader):
        z = model.encoder(x.to(device)).to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
    plt.colorbar()
    file_path = os.path.join(save_dir_path,'latent_space.png')
    if os.path.exists(file_path) : os.remove(file_path)
    plt.savefig(file_path)


def plot_latent_with_tsne(model, data_loader, save_dir_path, dim=2):
    plt.clf()

    latents = torch.Tensor()
    labels = torch.Tensor()
    for i, (x, y) in enumerate(data_loader):
        # z = model.encoder(x.to(device)).to('cpu').detach().numpy()
        z = model.encoder(x.to(device))
        latents = torch.cat((latents,z.to('cpu')),0)
        labels = torch.cat((labels,y),0)

    z = latents.to('cpu').detach().numpy()
    y = labels.to('cpu').detach().numpy()

    tsne_vector = manifold.TSNE(
        n_components=dim, learning_rate="auto", perplexity=40, init="pca", random_state=0
    ).fit_transform(z)

    if dim == 2:
        plt.scatter(tsne_vector[:, 0], tsne_vector[:, 1], c=y, cmap='tab10')
    elif dim == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(tsne_vector[:, 0], tsne_vector[:, 1], tsne_vector[:, 2], c=y, cmap='tab10')

    file_path = os.path.join(save_dir_path,'latent_space_with_tsne_{}d.png'.format(dim))
    if os.path.exists(file_path) : os.remove(file_path)
    plt.savefig(file_path)


def plot_reconstructed(model, image_size, num_channel, save_dir_path, r0=(-3, 3), r1=(-3, 3), n=15):
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


def plot_random_sample_from_prior(model, latent_dims, save_dir_path):
    z = torch.randn(10, latent_dims)
    samples = model.decoder(z.to(device))

    plt.clf()
    fig, axs = plt.subplots(1, 10, figsize=(16, 3))
    for j, sample in enumerate(samples):
        axs[j].imshow(sample.permute(1, 2, 0).to('cpu').detach().numpy())
        axs[j].axis('off')
    plt.tight_layout()
    plt.show()

    file_path = os.path.join(save_dir_path,'sample_from_prior.png')
    if os.path.exists(file_path) : os.remove(file_path)
    plt.savefig(file_path)

    return samples


def plot_reconstruct_from_images(model, data_loader, save_dir_path):
    og_images = None
    recon_images = None
    for i, (x,y) in enumerate(data_loader):
        og_images = x[:10]
        recon_images = model(x.to(device))[:10]
        break;

    plt.clf()
    fig, axs = plt.subplots(2, 10, figsize=(16, 6))
    for j, (og, recon) in enumerate(zip(og_images, recon_images)):
        axs[0, j].imshow(og.permute(1, 2, 0).to('cpu').detach().numpy())
        axs[1, j].imshow(recon.permute(1, 2, 0).to('cpu').detach().numpy())
        axs[0, j].axis('off')
        axs[1, j].axis('off')
    plt.tight_layout()
    plt.show()

    file_path = os.path.join(save_dir_path,'reconstruct_from_images.png')
    if os.path.exists(file_path) : os.remove(file_path)
    plt.savefig(file_path)


def interpolate(model, image_size, num_channel, save_dir_path, x1, x2, n=20):
    z1 = model.encoder(x1)[0]
    z2 = model.encoder(x2)[0]

    z = torch.stack([z1 + (z2 - z1)*t for t in np.linspace(0, 1, n)])
    interpolate_list = model.decoder(z.to(device)).to('cpu').detach()

    plt.clf()
    w = image_size
    img = np.zeros((w, n*w, num_channel))
    for i, x_hat in enumerate(interpolate_list):
        x_hat = x_hat.reshape(num_channel, image_size, image_size)
        recon_img = x_hat.permute(1, 2, 0).numpy()
        img[:, i*w:(i+1)*w, :] = recon_img
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])

    file_path = os.path.join(save_dir_path,'interpolated_images.png')
    if os.path.exists(file_path) : os.remove(file_path)
    plt.savefig(file_path)


def interpolate_gif(model, image_size, num_channel, save_dir_path, x1, x2, n=100):
    z1 = model.encoder(x1)[0]
    z2 = model.encoder(x2)[0]

    z = torch.stack([z1 + (z2 - z1)*t for t in np.linspace(0, 1, n)])
    interpolate_list = model.decoder(z.to(device)).to('cpu').detach() * 255

    mode = 'F' if num_channel == 1 else 'RGB'
    trans = transforms.ToPILImage(mode)
    images_list = []
    for x_hat in interpolate_list:
        recon_img = x_hat.reshape(num_channel, image_size, image_size)
        img = (trans(recon_img)).resize((256, 256))
        # recon_img = x_hat.reshape(num_channel, image_size, image_size)[0]
        # img2 = Image.fromarray(recon_img.numpy()).resize((256, 256))
        images_list.append(img)
    images_list = images_list + images_list[::-1] # loop back beginning

    plt.clf()
    file_path = os.path.join(save_dir_path,'interpolated_images.gif')
    if os.path.exists(file_path) : os.remove(file_path)

    images_list[0].save(
        file_path,
        save_all=True,
        append_images=images_list[1:],
        loop=1)


if __name__ == "__main__":
    main()