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
from dataset import *
from MNIST_classifier import *
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
nepochs = 100
beta1 = 0.5 # 'beta1 for adam. default=0.5'

parser = argparse.ArgumentParser()
parser.add_argument('--source_root', default='~/datasets', help='path to dataset MNIST')
parser.add_argument('--cuda_device', default='4', help='available cuda device.')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--classifier_M', default='./trained-models/MNIST_classifier.pth', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./trained-models', help='folder to output model checkpoints')
parser.add_argument('--outp', default='./fake-imgs-vanillaGAN', help='folder to output images')
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

dataset = Stacked_MNIST(imageSize=imageSize)
#dataset = Stacked_MNIST(load=False, source_root=opt.source_root, imageSize=imageSize)
nc=3

assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                         shuffle=True, num_workers=int(workers))


os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda_device
ngpu = len(opt.cuda_device.split(','))
device = torch.device("cuda:0")

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def generate_sample(generator, latent_size, num_image=20000, batch_size=100): #generate data sample to compute the fid.
    generator.eval()
    z_try = torch.randn(1, latent_size, 1, 1).to(device)
    data_try = generator(z_try)
    data_sample = torch.empty((num_image, data_try.shape[1], data_try.shape[2], data_try.shape[3]))

    for i in range(0, num_image, batch_size):
        start = i
        end = i + batch_size
        z = torch.randn(batch_size, latent_size, 1, 1).to(device)
        d = generator(z)
        data_sample[start:end] = d.cpu().data
    
    return data_sample

def compute_score(data, classifer):
    classifer = classifer.cuda()
    targets = np.zeros(1000, dtype=np.int32)
    for i in range(len(data)):
        y = np.zeros(3, dtype=np.int32)
        for j in range(3):#R, G, B
            x = data[i, j, :, :]
            x = torch.unsqueeze(x, dim=0)
            x = torch.unsqueeze(x, dim=0)
            x = x.cuda()
            output = classifer(x)
            predict = output.cpu().detach().max(1)[1]
            y[j] = predict
        result = 100 * y[0] + 10 * y[1] + y[2]
        targets[result] += 1
    
    covered_targets = np.sum(targets != 0)
    Kl_score = 0
    for i in range(1000):
        if targets[i] != 0:
            q = targets[i] / len(data)
            Kl_score +=  q * np.log(q * 1000)
    return covered_targets, Kl_score 

            


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

classifier_M = MLP(imageSize * imageSize, 10, [1024, 1024, 1024])
classifier_M.load_state_dict(torch.load(opt.classifier_M))

# encoder = torch.load('./sim_encoder.pth')
# encoder.eval()

criterion = nn.BCELoss()

fixed_noise = torch.randn(batchSize, nz, 1, 1, device=device)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

Kl_record = []
covered_targets_record = []

for epoch in range(nepochs):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        # netD.zero_grad()
        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)
        real_label = torch.full((batch_size,), 1, device=device)

        output = netD(real_cpu)
        errD_real = criterion(output, real_label)
        # errD_real.backward()
        D_x = output.mean().item()

        # inference the latent variable

        # train with fake
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(noise)
        fake_label = torch.full((batch_size,), 0, device=device)
        output = netD(fake.detach())
        errD_fake = criterion(output, fake_label)
        # errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        output = netD(fake)
        errG = criterion(output, real_label) # fake labels are real for generator cost
        if errG.item() < 3.2:
            optimizerD.zero_grad()
            errD.backward()
            optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        errG.backward()
        optimizerG.step()

        if i % 100 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f'
            % (epoch + 1, nepochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1))
    
    if (epoch + 1) % 10 == 0:
        vutils.save_image(real_cpu, '%s/real_samples.png' % opt.outp, normalize=True)
        fake = netG(fixed_noise)
        vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % (opt.outp, epoch + 1), normalize=True)

        dataset_fake = generate_sample(generator = netG, latent_size = nz)
        covered_targets, Kl_score = compute_score(dataset_fake, classifier_M)
        covered_targets_record.append(covered_targets)
        Kl_record.append(Kl_score)
        print("Covered Targets:{}, KL Score:{}".format(covered_targets, Kl_score))
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_vanilla_epoch_%d.pth' % (opt.outf, epoch + 1))
        torch.save(netD.state_dict(), '%s/netD_vanilla_epoch_%d.pth' % (opt.outf, epoch + 1))

with open('./score_record_vanilla.txt', 'w') as f:
    i0 = 0
    for (i, K) in zip(covered_targets_record, Kl_record):
        i0 += 1
        f.write("epoch " + str(10 * i0) + ":\n")
        f.write("covered targets:"+ str(i) + '\n')
        f.write("KL div:" + str(K) + '\n')