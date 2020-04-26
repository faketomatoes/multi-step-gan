from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from model_GAN import *
from fid import *
# from simple_ae import Encoder

from IPython import embed

lr = 0.0002
nz = 100 # size of latent variable
ngf = 64 
ndf = 64 
nef = 16

batchSize = 64
imageSize = 64 # 'the height / width of the input image to network'
workers = 2 # 'number of data loading workers'
nepochs = 300
beta1 = 0.5 # 'beta1 for adam. default=0.5'

#labels for the LS loss
a = 1
b = -1
c = 0
itfr_sigma = {0: 0.05, 50: 0.01, 100: 5e-3, 150: 1e-3, 200: 1e-4, 250: 3e-5}
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10', help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', default='~/datasets/data_cifar10', help='path to dataset')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./trained_model', help='folder to output model checkpoints')
parser.add_argument('--outp', default='./try', help='folder to output images')
parser.add_argument('--manualSeed', type=int, help='manual random seed')
parser.add_argument('--classes', default='bedroom', help='comma separated list of classes for the lsun data set')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(imageSize),
                                   transforms.CenterCrop(imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    nc=3
elif opt.dataset == 'lsun':
    classes = [ l + '_train' for l in opt.classes.split(',')]
    dataset = dset.LSUN(root=opt.dataroot, classes=classes,
                        transform=transforms.Compose([
                            transforms.Resize(imageSize),
                            transforms.CenterCrop(imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
    nc=3
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
    nc=3
    m_true, s_true = compute_cifar10_statistics()

elif opt.dataset == 'mnist':
        dataset = dset.MNIST(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,), (0.5,)),
                           ]))
        nc=1

elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, imageSize, imageSize),
                            transform=transforms.ToTensor())
    nc=3

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                         shuffle=True, num_workers=int(workers))

device = torch.device("cuda:1")
ngpu = 1

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def generate_sample(generator, latent_size, num_image=10000, batch_size=50): #generate data sample to compute the fid.
    generator.eval()
    z_try = Variable(torch.randn(1, latent_size, 1, 1).to(device))
    data_try = generator(z_try)

    data_sample = np.empty((num_image, data_try.shape[1], data_try.shape[2], data_try.shape[3]))

    for i in range(0, num_image, batch_size):
        start = i
        end = i + batch_size
        z = Variable(torch.randn(batch_size, latent_size, 1, 1).to(device))
        d = generator(z)
        data_sample[start:end] = d.cpu().data.numpy()
    
    return data_sample

netG = Generator(ngpu, nz, ngf, nc).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = Discriminator(ngpu, ndf, nc).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# encoder = torch.load('./sim_encoder.pth')
# encoder.eval()

criterion = nn.BCELoss()

fixed_noise = torch.randn(batchSize, nz, 1, 1, device=device)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

fid_record = []

for epoch in range(nepochs):
    if epoch in itfr_sigma:
        sigma = itfr_sigma[epoch]
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        # netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        real_label = torch.full((batch_size,), 1, device=device)
        pixd_noise = torch.randn(real_cpu.size(), device=device)
        pixg_noise = torch.randn(real_cpu.size(), device=device)
        output = netD(real_cpu + sigma * pixd_noise)
        errD_real = torch.mean((output - a) ** 2)
        # errD_real.backward()
        D_x = torch.sigmoid(output).mean().item()

        # inference the latent variable

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        fake_label = torch.full((batch_size,), 0, device=device)
        output = netD(netG(noise) + sigma * pixg_noise)[0]
        errD_fake = torch.mean((output - b) ** 2)
        # errD_fake.backward()
        D_G_z1 = torch.sigmoid(output).mean().item()
        errD = errD_real + errD_fake
        output = netD(fake)
        errG = torch.mean((output - c) ** 2) # fake labels are real for generator cost

        optimizerD.zero_grad()
        errD.backward(retain_graph=True)
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        errG.backward()
        optimizerG.step()

        if i % 100 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f'
            % (epoch, nepochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1))
    
    if (epoch + 1) % 10 == 0:
        vutils.save_image(real_cpu, '%s/real_samples.png' % opt.outp, normalize=True)
        fake = netG(fixed_noise)
        vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % (opt.outp, epoch + 1), normalize=True)

        dataset_fake = generate_sample(generator = netG, latent_size = nz)
        fid = calculate_fid(dataset_fake, m_true, s_true)
        fid_record.append(fid)
        print("The Frechet Inception Distance:", fid)
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch + 1))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch + 1))

with open('./fid_record.txt', 'w') as f:
    for i in fid_record:
        f.write(str(i) + '\n')