import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, nz, nef, nc):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, nef, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef) x 32 x 32
            nn.Conv2d(nef, nef * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*2) x 16 x 16
            nn.Conv2d(nef * 2, nef * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*4) x 8 x 8
            nn.Conv2d(nef * 4, nef * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(nef * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (nef*8) x 4 x 4
            nn.Conv2d(nef * 8, nz, 4, 1, 0, bias=False),
            # nn.Sigmoid()
            nn.BatchNorm2d(nz)
        )
    
    def forward(self, x):
        x = self.model(x)
        return x

class Decoder(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
        # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
            )

    def forward(self, x):
            x = self.model(x)
            return x


# class autoencoder(nn.Module):
#     def __init__(self):
#         super(autoencoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(28 * 28, 128),
#             nn.ReLU(True),
#             nn.Linear(128, 64),
#             nn.ReLU(True), nn.Linear(64, 12), nn.ReLU(True), nn.Linear(12, 3))
#         self.decoder = nn.Sequential(
#             nn.Linear(3, 12),
#             nn.ReLU(True),
#             nn.Linear(12, 64),
#             nn.ReLU(True),
#             nn.Linear(64, 128),
#             nn.ReLU(True), nn.Linear(128, 28 * 28), nn.Tanh())

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x