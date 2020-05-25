import torch
import sys
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import plotly.express as px
import plotly.io as pio

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.optim import Adam
from sklearn.manifold import TSNE

home = os.environ["HOME"]
sys.path.append(os.path.join(home, "Desktop/productivity_efficiency/torch_trainer"))

from trainer import Trainer

class MNISTDataset(MNIST):

    def __getitem__(self, index):
        img, _ = super(MNISTDataset, self).__getitem__(index)
        return img, img

class Encoder(nn.Module):

    def __init__(self, z_dim, fc1_size):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.bn_conv1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.bn_conv2 = nn.BatchNorm2d(32)
        
        
        self.fc1 = nn.Linear(fc1_size, 400)
        self.bn_fc1 = nn.BatchNorm1d(400)
        self.z_mu = nn.Linear(400, z_dim)
        self.z_sigma = nn.Linear(400, z_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn_conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn_conv2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn_fc1(x)
        z_loc = self.z_mu(x)
        z_scale = self.z_sigma(x)

        return z_loc, z_scale

class Decoder(nn.Module):

    def __init__(self, z_dim, output_size):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(z_dim, 400)
        self.bn_fc1 = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400, output_size)
        self.bn_fc2 = nn.BatchNorm1d(output_size)

        self.conv1 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn_conv1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 1, 3, 1, 1)
    
    def forward(self, z_input):
        x = self.fc1(z_input)
        x = F.relu(x)
        x = self.bn_fc1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn_fc2(x)
        
        x = x.view(z_input.size()[0], 32, 7, 7)
        x = F.interpolate(x, (14, 14))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn_conv1(x)
        x = F.interpolate(x, (28, 28))
        x = self.conv2(x)
        output = torch.sigmoid(x)

        return output

class VAE(nn.Module):

    def __init__(self, z_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(z_dim, 1568)
        self.decoder = Decoder(z_dim, 1568)

    def forward(self, x):

        z_mean, z_sigma = self.encoder(x)
        std = torch.exp(0.5*z_sigma)
        eps = torch.randn_like(std)
        output = self.decoder(z_mean+eps*std)

        return output, z_mean, z_sigma

    def reconstruct_digit(self, sample):
        return self.decoder(sample)

def loss_fn(output, mean, logvar, target):
    bce = F.binary_cross_entropy(output, target, reduction='sum')
    
    kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    return bce + kl

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MNIST VAE")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--z_dim", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--print_freq", type=int, default=10)
    parser.add_argument("--weights", type=str, default="vae.pth")
    parser.add_argument("--visualise", type=bool, default=True)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)

    vae = VAE(args.z_dim)
    optimizer = Adam(vae.parameters(), lr=args.lr)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    vae.to(device)
    # download mnist & setup loaders
    if args.mode == "train":
        train_loader = DataLoader(MNISTDataset('./data', train=True, download=True, transform=transforms.ToTensor()),
            batch_size=128, shuffle=True)
        val_loader = DataLoader(MNISTDataset('./data', train=False, transform=transforms.ToTensor()),
            batch_size=128, shuffle=True)

        trainer = Trainer(vae, args.epochs, train_loader, val_loader, device, loss_fn, optimizer, args.print_freq)
        trainer.train_model()
        torch.save(vae.state_dict(), args.weights)

    val_loader = DataLoader(MNIST('./data', train=False, transform=transforms.ToTensor()),
        batch_size=128, shuffle=True)
    
    if args.mode != "train":
        vae.load_state_dict(torch.load(args.weights))
        vae.eval()
    
    if args.visualise:

        latent_mnist = []
        target = []
        for data, targets in val_loader:
            latent_means, latent_sigma = vae.encoder(data)
            latent_mnist.extend(latent_means.detach().numpy())
            target.extend(targets.numpy())

        # take first 1k
        latent = np.array(latent_mnist[:1000])
        target = np.array(target[:1000])
        tsne = TSNE(n_components=2, init="pca", random_state=0)

        X = tsne.fit_transform(latent)

        data = np.vstack((X.T, target)).T
        df = pd.DataFrame(data=data, columns=["z1", "z2", "label"])
        df["label"] = df["label"].astype(str)

        fig = px.scatter(df, x="z1", y="z2", color="label")

        pio.write_html(fig, file="raw.html", auto_open=True)