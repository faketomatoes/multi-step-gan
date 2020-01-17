import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import torchvision.datasets as dset
from torchvision.utils import save_image

if not os.path.exists('./mlp_img'):
    os.mkdir('./mlp_img')


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3, 64, 64)
    return x


num_epochs = 100
batch_size = 128
learning_rate = 1e-3
nef = 16
ngf = 16
nc = 3
nz = 100
imageSize = 64
dataroot = '/data1/liangjiadong/cifar-10-batches-py'

# img_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# dataset = MNIST('~/flow_simple/mnist_dset', transform=img_transform, download= True)
dataset = dset.CIFAR10(
    root=dataroot, download=True,
    transform=transforms.Compose(
        [transforms.Resize(imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    ))
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


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


class Encoder(nn.Module):
    def __init__(self):
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
    def __init__(self):
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


# model = autoencoder().cuda()
encoder = Encoder().cuda()
decoder = Decoder().cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    [{'params':encoder.parameters()}, {'params':decoder.parameters()}], lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        # img = img.view(img.size(0), nc, )
        img = Variable(img).cuda()
        # ===================forward=====================
        output = encoder(img)
        output = decoder(output)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.item()))
    if epoch % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, './mlp_img/image_{}.png'.format(epoch))

torch.save(encoder.state_dict(), './sim_encoder.pth')
torch.save(decoder.state_dict(), './sim_decoder.pth')