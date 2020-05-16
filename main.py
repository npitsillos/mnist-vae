import torch
import sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.distributions.normal import Normal
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from sklearn.manifold import TSNE

sys.path.append("/home/pitsillos/Desktop/productivity_efficiency/torch_trainer")

from trainer import Trainer

# class MNISTDataset(MNIST):

#     def __getitem__(self, index):
#         img, _ = super(MNISTDataset, self).__getitem__(index)
#         return img, img

class Encoder(nn.Module):

    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.bn_conv1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.bn_conv2 = nn.BatchNorm2d(32)
        
        self.fc1 = nn.Linear(1568, 400)
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

    def __init__(self, z_dim):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(z_dim, 400)
        self.bn_fc1 = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400, 1568)
        self.bn_fc2 = nn.BatchNorm1d(1568)

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
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

    def forward(self, x):

        z_mean, z_sigma = self.encoder(x)
        std = torch.exp(0.5*z_sigma)
        eps = torch.randn_like(std)
        output = self.decoder(z_mean+eps*std)

        return output, z_mean, z_sigma

def loss_fn(output, target):
    bce = F.binary_cross_entropy(output[0], target, reduction='sum')
    
    kl = -0.5 * torch.sum(1 + output[2] - output[1].pow(2) - output[2].exp())

    return bce + kl

torch.manual_seed(1)
# download mnist & setup loaders
train_loader = DataLoader(MNIST('./data', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=128, shuffle=True)
val_loader = DataLoader(MNIST('./data', train=False, transform=transforms.ToTensor()),
    batch_size=128, shuffle=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
vae = VAE(20)

# optimizer = Adam(vae.parameters(), lr=1e-3)

# trainer = Trainer(vae, 10, train_loader, val_loader, device, loss_fn, optimizer, 10)
# trainer.train_model()
# torch.save(vae.state_dict(), "./vae.pth")

vae.to(device)
vae.load_state_dict(torch.load("./vae.pth"))
vae.eval()
latent_mnist = []
target = []
for data, targets in val_loader:
    latent_means, latent_sigma = vae.encoder(data)
    latent_mnist.extend(latent_means.detach().numpy())
    target.extend(targets.numpy())

# take first 1k
latent = np.array(latent_mnist)
target = np.array(target)
tsne = TSNE(n_components=2, init="pca", random_state=0)

X = tsne.fit_transform(latent)

data = np.vstack((X.T, target)).T
df = pd.DataFrame(data=data, columns=["z1", "z2", "label"])

sns.FacetGrid(df, hue="label", size=6).map(plt.scatter, "z1", "z2").add_legend()
plt.show()