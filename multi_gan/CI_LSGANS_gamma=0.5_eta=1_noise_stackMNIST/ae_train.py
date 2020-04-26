import os

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as dset
from torchvision.utils import save_image

import argparse
import numpy as np

from model_ae import *
from dataset import *


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

parser = argparse.ArgumentParser()
parser.add_argument('--source_root', default="~/datasets")
parser.add_argument('--cuda_device', default="0")
parser.add_argument('--save_model_dir', default="./trained-models")
parser.add_argument('--save_image_dir', default='./mlp_img')
opt = parser.parse_args()
print(opt)


os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda_device
ngpu = len(opt.cuda_device.split(','))
if not os.path.exists(opt.save_image_dir):
    os.mkdir(opt.save_image_dir)

# img_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])
# dataset = Stacked_MNIST(load=False, source_root=opt.source_root)
dataset = Stacked_MNIST()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
# model = autoencoder().cuda()
encoder = Encoder(ngpu=ngpu, nc=nc, nef=nef, nz=nz).cuda()
decoder = Decoder(nc=nc, ngf=ngf, nz=nz).cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    [{'params':encoder.parameters()}, {'params':decoder.parameters()}], lr=learning_rate, weight_decay=1e-5)

for epoch in range(num_epochs):
    for data in dataloader:
        img, _ = data
        # img = img.view(img.size(0), nc, )
        img = img.cuda()
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
    if (epoch + 1) % 10 == 0:
        pic = to_img(output.cpu().data)
        save_image(pic, os.path.join(opt.save_image_dir, 'image_{}.png'.format(epoch + 1)))

torch.save(encoder.state_dict(), os.path.join(opt.save_model_dir, 'sim_encoder.pth'))
torch.save(decoder.state_dict(), os.path.join(opt.save_model_dir, 'sim_decoder.pth'))