from __future__ import print_function
import argparse
import math
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
import numpy
from model_GAN import *
from model_ae import *
from fid import *
# from simple_ae import Encoder

from IPython import embed

nz = 100 # size of latent variable
ngf = 64 
ndf = 64 
nef = 16
np = 4
width = 25 # width = nz / np
#label for LSGAN loss
a = 1
b = -1
c = 0
itfr_sigma = {0: 0.05, 50: 0.01, 100: 1e-3, 150: 1e-4, 250: 3e-5}

lr = 0.0002
lr_encoder = 0.01
batchSize = 32
imageSize = 64 # 'the height / width of the input image to network'
workers = 2 # 'number of data loading workers'
nepochs = 300
beta1 = 0.5 # 'beta1 for adam. default=0.5'
weight_decay_coeff = 5e-4 # weight decay coefficient for training netE.
alpha = 0.5 # coefficient for GAN_loss tern when training netE
gamma = 0.2 # coefficient for the mutual information
eta = 0.5 # coefficient for the reconstruction err when training E
default_device = 'cuda:0'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='cifar10', help='cifar10 | lsun | mnist |imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', default='~/datasets/data_cifar10', help='path to dataset')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netE', default='', help='path to netE.')
parser.add_argument('--outf', default='./trained_model', help='folder to output model checkpoints')
parser.add_argument('--outp', default='./fake-imgs', help='folder to output images')
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
    classes = [ c + '_train' for c in opt.classes.split(',')]
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


ngpu = 1
device = torch.device(default_device)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1 and not (m.weight is None):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Affine') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def generate_sample(generator, latent_size, num_image=50000, batch_size=50): #generate data sample to compute the fid.
    generator.eval()
    
    z_try = Variable(torch.randn(1, latent_size, 1, 1).to(device))
    data_try = generator(z_try)

    data_sample = numpy.empty((num_image, data_try.shape[1], data_try.shape[2], data_try.shape[3]))
    for i in range(0, num_image, batch_size):
        start = i
        end = i + batch_size
        z = Variable(torch.randn(batch_size, latent_size, 1, 1).to(device))
        d = generator(z)
        data_sample[start:end] = d.cpu().data.numpy()
    
    return data_sample


netE = Encoder(ngpu, nz, nef, nc).to(device)
netE.apply(weights_init)
if opt.netE != '':
    netE.load_state_dict(torch.load(opt.netE))
print(netE)
# encoder = torch.load('./sim_encoder.pth')


netG = Generator(ngpu, nz, ngf, nc).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = Discriminator(ngpu, ndf, nc, np).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# encoder = torch.load('./sim_encoder.pth')
# encoder.eval()


criterion_reconstruct = nn.L1Loss()
criterion = nn.CrossEntropyLoss()

fixed_noise = torch.randn(batchSize, nz, 1, 1, device=device)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerE = optim.Adam(netE.parameters(), lr=lr_encoder, betas=(beta1, 0.999), weight_decay=weight_decay_coeff)

fid_record = []


for epoch in range(nepochs):
    if epoch in itfr_sigma:
        sigma = itfr_sigma[epoch]
    for i, data in enumerate(dataloader, 0):
        # netG.train()
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        # netD.zero_grad()
        real = data[0].to(device)
        batch_size = real.size(0)
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        latent_real = netE(real)

        pixd_noise = torch.randn(real.size(), device=device)
        pixg_noise = torch.randn(real.size(), device=device)
        
        label_real = torch.full((batch_size,), 0, device=device, dtype=torch.int64)
        output = netD(real + sigma * pixd_noise)[0] # unnormalized
        errD = torch.mean((output - a) ** 2)
        D_x = torch.sigmoid(output).mean().item()
        label_fake = torch.full((batch_size,), 1, device=device, dtype=torch.int64)
        output = netD(netG(noise) + sigma * pixg_noise)[0]
        errD += torch.mean((output - b) ** 2)
        D_Gz = torch.sigmoid(output).mean().item()

        errG = torch.mean((output - c) ** 2) + 0.0

        k = torch.randint(np, (1,), dtype=torch.int64).item()
        noise_mask = torch.zeros((batch_size, nz, 1, 1), device=device)
        real_mask = torch.ones((batch_size, nz, 1, 1), device=device)
        index = torch.tensor(range(k * width, (k + 1) * width), dtype=torch.int64, device=device)
        noise_mask = noise_mask.index_fill_(1, index, 1)
        real_mask = real_mask.index_fill_(1, index, 0)
        latent = torch.mul(latent_real, real_mask) + torch.mul(noise, noise_mask)
        fake = netG(latent)
        label = torch.full((batch_size,), k, device=device, dtype=torch.int64)
        output = netD(fake)[1]
        CE_regularizer = gamma * criterion(output, label)
        errD += CE_regularizer
        errG -= CE_regularizer
        errG += eta * criterion_reconstruct(real, netG(latent_real))
        
        optimizerD.zero_grad()
        errD.backward(retain_graph=True)
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        optimizerG.zero_grad()
        errG.backward()
        optimizerG.step()

        ############################
        # (3) Update E network: minimize reconstruction error
        ###########################

        netE.train()
        for j in range(10):
            err_reconstruct = criterion_reconstruct(real, netG(netE(real)))
            GAN_loss = torch.tensor(0.0, device=device)
            noise_mask = torch.zeros((batch_size, nz, 1, 1), device=device)
            real_mask = torch.ones((batch_size, nz, 1, 1), device=device)
            index = torch.tensor(range(k * width, (k + 1) * width), dtype=torch.int64, device=device)
            noise_mask = noise_mask.index_fill_(1, index, 1)
            real_mask = real_mask.index_fill_(1, index, 0)
            latent = torch.mul(netE(real), real_mask) + torch.mul(noise, noise_mask)
            fake = netG(latent)
            label = torch.full((batch_size,), k, device=device, dtype=torch.int64)
            output = netD(fake)[1]
            GAN_loss += gamma * criterion(output, label)
            errE = err_reconstruct + alpha * GAN_loss
            optimizerE.zero_grad()
            errE.backward()
            optimizerE.step()
        netE.eval()
        


        if i % 500 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x):%.4f D(G(z)):%.4f CE_regularizer: %.4f Reconstruct_err: %.4f'
            % (epoch, nepochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_Gz, 0 - gamma * CE_regularizer.item(), err_reconstruct))
    
    if (epoch + 1) % 10 == 0:
        netG.eval()
        vutils.save_image(real, '%s/real_samples.png' % opt.outp, normalize=True)
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
