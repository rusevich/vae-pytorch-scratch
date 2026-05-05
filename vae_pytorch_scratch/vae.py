import numpy as np
import torch
import torch.nn as nn


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
        eps = torch.normal(0, 1, size=(batch_size, self.latent_dim))
        z = mu + eps * std

        # decode
        out = self.decoder(z)

        return out, mu, std


class ELBOLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y, y_prim, mu, std):
        mse = torch.nn.MSELoss()
        l2 = torch.sqrt(mse(y, y_prim))
        kl = torch.mean(-0.5 * torch.sum(1 + torch.log(std**2) - mu**2 - std**2, dim=1))

        return l2, kl
