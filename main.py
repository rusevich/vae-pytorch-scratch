import os
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torchvision
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# DATA PREP

plt.style.use('dark_background')

EXAMPLES_FOR_PLOT = 5 # should be less than batch

LATENT_SIZE = 64
BATCH_SIZE = 64
EPOCHS = 100
DATA_PATH = "data"
DEVICE = "cpu"

if torch.cuda.is_available():
    DEVICE = "cuda"

os.makedirs(DATA_PATH, exist_ok=True)
transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    # v2.Normalize((0.1307,), (0.3081,)) TODO investigate
])
train_data = torchvision.datasets.MNIST(DATA_PATH, download=True, transform=transform, train=True)
test_data = torchvision.datasets.MNIST(DATA_PATH, download=True, transform=transform, train=False)

train_dataloader = DataLoader(train_data, BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, BATCH_SIZE, shuffle=False)

# getting EXAMPLES_FOR_PLOT examples for plotting, TODO rewrite
examples = None
for X, _ in test_dataloader:
    examples = X[:EXAMPLES_FOR_PLOT, :].to(DEVICE)
    break

# MODEL

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 28 -> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
        )

        self.mu_head = nn.Linear(256, latent_dim)
        self.std_head = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (128, 7, 7)),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 7 -> 14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),  # 14 -> 28
            nn.Sigmoid(),
        )

    def forward(self, x):
        # encode
        x = self.encoder(x)
        mu = self.mu_head(x)
        std = self.std_head(x)

        # reparametrization trick
        batch_size = x.size(0)
        eps = torch.normal(0, 1, size=(batch_size, self.latent_dim)).to(DEVICE)
        z = mu + eps * std

        # decode
        out = self.decoder(z)

        return out, mu, std

class ELBOLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, y_prim, mu, std):
        mse = torch.nn.MSELoss()
        l2 = mse(y, y_prim) # l2 = F.binary_cross_entropy(y, y_prim, reduction='sum') / BATCH_SIZE
        kl = torch.mean(-0.5 * torch.sum(1 + torch.log(std**2) - mu**2 - std**2, dim=1))

        return l2, kl

# TRAIN LOOP

model = VAE(LATENT_SIZE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())
loss_fn = ELBOLoss()

metrics = defaultdict(list)
# train_l2, train_kl, train_loss
# test_l2, test_kl, test_loss

for epoch in range(1, EPOCHS):
    metrics_epoch = defaultdict(list)

    # train
    model.train()
    for X, _ in train_dataloader:
        X = X.to(DEVICE)
        X_prim, mu, std = model.forward(X)
        l2, kl = loss_fn(X, X_prim, mu, std)
        loss = l2 + kl

        loss.backward()
        optimizer.step()
        model.zero_grad()

        # save losses
        metrics_epoch["train_l2"].append(l2.item())
        metrics_epoch["train_kl"].append(kl.item())
        metrics_epoch["train_loss"].append(loss.item())

    # test
    model.eval()
    with torch.no_grad():
        for X, _ in test_dataloader:
            X = X.to(DEVICE)
            X_prim, mu, std = model.forward(X)

            l2, kl = loss_fn(X, X_prim, mu, std)
            loss = l2 + kl

            # save losses
            metrics_epoch["test_l2"].append(l2.item())
            metrics_epoch["test_kl"].append(kl.item())
            metrics_epoch["test_loss"].append(loss.item())

    # update metrics dict
    for key, value in metrics_epoch.items():
        mean_value = np.mean(value)
        metrics[key].append(mean_value)

    print(f"--------- ITERATION {epoch} ---------")
    print(f"train loss: {metrics["train_loss"][-1]}")
    print(f"train kl: {metrics["train_kl"][-1]}")
    print(f"train l2: {metrics["train_l2"][-1]}")
    print()
    print(f"test loss: {metrics["test_loss"][-1]}")
    print(f"test kl: {metrics["test_kl"][-1]}")
    print(f"test l2: {metrics["test_l2"][-1]}")

    # plot
    plt.clf()
    x = list(range(epoch))

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(5, 5)

    # losses graph
    plt.subplot(gs[0:3, :])
    plt.plot(x, metrics["train_loss"], color="red", label="train_loss")
    plt.plot(x, metrics["train_kl"], color="orange", label="train_kl")
    plt.plot(x, metrics["train_l2"], color="yellow", label="train_l2")

    plt.plot(x, metrics["test_loss"], color="green", label="test_loss")
    plt.plot(x, metrics["test_kl"], color="blue", label="test_kl")
    plt.plot(x, metrics["test_l2"], color="brown", label="test_l2")

    plt.legend()

    # images examples
    examples_prim, _, _ = model.forward(examples)
    for i in range(EXAMPLES_FOR_PLOT):
        ex = examples[i, :].cpu().squeeze().detach().numpy()
        ex_prim = examples_prim[i, :].cpu().squeeze().detach().numpy()
        plt.subplot(gs[3, i])
        plt.axis("off")
        plt.imshow(ex, cmap="gray")
        plt.subplot(gs[4, i])
        plt.axis("off")
        plt.imshow(ex_prim, cmap="gray")

    plt.show()
