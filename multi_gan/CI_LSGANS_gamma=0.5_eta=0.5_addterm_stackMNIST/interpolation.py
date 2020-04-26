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
import numpy
from model_GAN import *
from model_ae import *
from dataset import *
from MNIST_classifier import *
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
itfr_sigma = {0: 5e-2, 100: 1e-2}

lr = 0.0002
lr_encoder = 0.01
batchSize = 32
imageSize = 64 # 'the height / width of the input image to network'
workers = 2 # 'number of data loading workers'
nepochs = 150
beta1 = 0.5 # 'beta1 for adam. default=0.5'
weight_decay_coeff = 5e-4 # weight decay coefficient for training netE.
alpha = 0.2 # coefficient for GAN_loss term when training netE
gamma = 0.5 # coefficient for the mutual information
eta = 0.5 # coefficient for the reconstruction err when training G
default_device = 'cuda:3'

parser = argparse.ArgumentParser()
parser.add_argument('--source_root', default='~/datasets', help='path to dataset MNIST')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netE', default='', help='path to netE.')
parser.add_argument('--classifier_M', default='./trained-models/MNIST_classifier.pth', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./trained-models', help='folder to output model checkpoints')
parser.add_argument('--outp', default='./fake-imgs-onestep', help='folder to output images')
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

#dataset = Stacked_MNIST(imageSize=imageSize)
dataset = Stacked_MNIST(load=True, source_root=opt.source_root, imageSize=imageSize)
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
    targets = numpy.zeros(1000, dtype=numpy.int32)
    for i in range(len(data)):
        y = numpy.zeros(3, dtype=numpy.int32)
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
    
    covered_targets = numpy.sum(targets != 0)
    Kl_score = 0
    for i in range(1000):
        if targets[i] != 0:
            q = targets[i] / len(data)
            Kl_score +=  q * numpy.log(q * 1000)
    return covered_targets, Kl_score 


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
classifier_M = MLP(imageSize * imageSize, 10, [1024, 1024, 1024])
classifier_M.load_state_dict(torch.load(opt.classifier_M))
# encoder = torch.load('./sim_encoder.pth')
# encoder.eval()

criterion = nn.CrossEntropyLoss()
criterion_reconstruct = nn.L1Loss()

fixed_noise = torch.randn(batchSize, nz, 1, 1, device=device)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerE = optim.Adam(netE.parameters(), lr=lr_encoder, betas=(beta1, 0.999), weight_decay=weight_decay_coeff)

Kl_record = []
covered_targets_record = []

for epoch in range(nepochs):
    # itpl = [num_inter - epoch, 0] # num_inter指的是进行插值的epoch数量
    # itpl_vl = max(itpl)
    # itpl_vl = float(itpl_vl)
    # print("itpl_vl: %d" % itpl_vl)
    # 作为ablation study的一部分暂时去掉了插值部分
    if epoch in itfr_sigma:
        sigma = itfr_sigma[epoch]
    for i, data in enumerate(dataloader, 0):
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
        errG += eta * criterion_reconstruct(latent, netE(fake))
        
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
            err_reconstruct = criterion_reconstruct(real, netG(netE(real)))
            err_reconstruct += criterion_reconstruct(latent, netE(fake.detach()))
            errE = err_reconstruct + alpha * GAN_loss
            optimizerE.zero_grad()
            errE.backward()
            optimizerE.step()
        netE.eval()
        


        if i % 500 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x):%.4f D(G(z)):%.4f CE_regularizer: %.4f Reconstruct_err: %.4f'
            % (epoch, nepochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_Gz, 0 - gamma * CE_regularizer.item(), err_reconstruct))
    
    if (epoch + 1) % 10 == 0:
        vutils.save_image(real, '%s/real_samples.png' % opt.outp, normalize=True)
        fake = netG(fixed_noise)
        vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % (opt.outp, epoch + 1), normalize=True)

        dataset_fake = generate_sample(generator = netG, latent_size = nz)
        covered_targets, Kl_score = compute_score(dataset_fake, classifier_M)
        covered_targets_record.append(covered_targets)
        Kl_record.append(Kl_score)
        print("Covered Targets:{}, KL Score:{}".format(covered_targets, Kl_score))
        torch.save(netG.state_dict(), './trained-models/final_netG.pth')
        torch.save(netE.state_dict(), './trained-models/final_netE.pth')

with open('./score_record_BCGAN.txt', 'w') as f:
    i0 = 0
    for (i, K) in zip(covered_targets_record, Kl_record):
        i0 += 1
        f.write("epoch " + str(10 * i0) + ":\n")
        f.write("covered targets:"+ str(i) + '\n')
        f.write("KL div:" + str(K) + '\n')