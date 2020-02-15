import argparse
import torch
from torchvision import datasets, transforms
import torch.optim as optim
from torch.autograd import Variable
import torchvision.utils as vutils
from model import *
from fid import *
import os

from tensorboardX import SummaryWriter

batch_size = 100
lr = 1e-4
latent_size = 256
num_epochs = 100
cuda_device = "0"
interpolation = 20


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='cifar10 | svhn', default="cifar10")
parser.add_argument('--dataroot', help='path to dataset', default="~/datasets/data_cifar10")
parser.add_argument('--use_cuda', type=boolean_string, default=True)
parser.add_argument('--save_model_dir', default="./try")
parser.add_argument('--save_image_dir', default="./try")
parser.add_argument('--reuse', type=boolean_string, default=False)
parser.add_argument('--save_freq', type=int, default=20)

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
print(opt)

def tocuda(x):
    if opt.use_cuda:
        return x.cuda()
    return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.bias.data.fill_(0)


def log_sum_exp(input):
    m, _ = torch.max(input, dim=1, keepdim=True)
    input0 = input - m
    m.squeeze()
    return m + torch.log(torch.sum(torch.exp(input0), dim=1))


def get_log_odds(raw_marginals):
    marginals = torch.clamp(raw_marginals.mean(dim=0), 1e-7, 1 - 1e-7)
    return torch.log(marginals / (1 - marginals))

def generate_sample(generator, latent_size, num_image=1000, batch_size=50): #generate data sample to compute the fid.
    generator.eval()
    
    z_try = Variable(tocuda(torch.randn(1, latent_size, 1, 1)))
    data_try = generator(z_try)

    data_sample = np.empty((num_image, data_try.shape[1], data_try.shape[2], data_try.shape[3]))

    for i in range(0, num_image, batch_size):
        start = i
        end = i + batch_size
        z = Variable(tocuda(torch.randn(batch_size, latent_size, 1, 1)))
        d = generator(z)
        data_sample[start:end] = d.cpu().data.numpy()
    
    return data_sample


if opt.dataset == 'svhn':
    train_loader = torch.utils.data.DataLoader(
        datasets.SVHN(root=opt.dataroot, split='extra', download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor()
                      ])),
        batch_size=batch_size, shuffle=True)
#    m_true, s_true = compute_svhn_statistics(batch_size=50, dims=2048, cuda=True)
elif opt.dataset == 'cifar10':
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=opt.dataroot, train=True, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor()
                      ])),
        batch_size=batch_size, shuffle=True)
    m_true, s_true = compute_cifar10_statistics(batch_size=50, dims=2048, cuda=True, data_root=opt.dataroot)
else:
    raise NotImplementedError

netE = tocuda(Encoder(latent_size, True))
netG = tocuda(Generator(latent_size))
netD = tocuda(Discriminator(latent_size, 0.2, 1))

netE.apply(weights_init)
netG.apply(weights_init)
netD.apply(weights_init)

optimizerG = optim.Adam([{'params' : netE.parameters()},
                         {'params' : netG.parameters()}], lr=lr, betas=(0.5,0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCELoss() #binary cross entropy
reconstruct = nn.L1Loss()

count = 0

for epoch in range(num_epochs):

    ip = [interpolation - epoch, 0] #interpolation？
    ip_num = max(ip)
    ip_num = ip_num/(2 * interpolation)

    writer = SummaryWriter('~/multistep_gan/runs/bigan_inter')

    i = 0
    for (data, target) in train_loader:

        count += 1

        real_label = Variable(tocuda(torch.ones(batch_size)))
        fake_label = Variable(tocuda(torch.zeros(batch_size)))

        noise1 = Variable(tocuda(torch.Tensor(data.size()).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs)))
        noise2 = Variable(tocuda(torch.Tensor(data.size()).normal_(0, 0.1 * (num_epochs - epoch) / num_epochs)))

        if epoch == 0 and i == 0:
            netG.output_bias.data = get_log_odds(tocuda(data))

        if data.size()[0] != batch_size:
            continue

        d_real = Variable(tocuda(data))

        z_fake = Variable(tocuda(torch.randn(batch_size, latent_size, 1, 1)))
        # d_fake = netG(z_fake)

        z_real, _, _, _ = netE(d_real)
        z_real = z_real.view(batch_size, -1)

        mu, log_sigma = z_real[:, :latent_size], z_real[:, latent_size:]
        sigma = torch.exp(log_sigma)
        epsilon = Variable(tocuda(torch.randn(batch_size, latent_size)))

        output_z = mu + epsilon * sigma
        rec_real = netG(output_z.view(batch_size, latent_size, 1, 1)) # ？

        z_fake = (1 - ip_num) * z_fake + ip_num * output_z.view(batch_size, latent_size, 1, 1)
        d_fake = netG(z_fake)

        output_real, _ = netD(d_real + noise1, output_z.view(batch_size, latent_size, 1, 1))
        output_fake, _ = netD(d_fake + noise2, z_fake)

        loss_d = criterion(output_real, real_label) + criterion(output_fake, fake_label)
        loss_g = criterion(output_fake, real_label) + criterion(output_real, fake_label) + 0.3 * reconstruct(rec_real, d_real)

        if loss_g.item() < 3.5:
            optimizerD.zero_grad()
            loss_d.backward(retain_graph=True)
            optimizerD.step()

        optimizerG.zero_grad()
        loss_g.backward()
        optimizerG.step()

        writer.add_scalar('D Loss', loss_d, global_step=count)
        writer.add_scalar('G Loss', loss_g, global_step=count)

        if i % 100 == 0:
            print("Epoch :", epoch, "Iter :", i, "D Loss :", loss_d.item(), "G loss :", loss_g.item(),
                  "D(x) :", output_real.mean().item(), "D(G(x)) :", output_fake.mean().item())

        if i % 50 == 0:
            vutils.save_image(d_fake.cpu().data[:16, ], './%s/fake.png' % (opt.save_image_dir))
            vutils.save_image(d_real.cpu().data[:16, ], './%s/real.png'% (opt.save_image_dir))

        i += 1

    if (epoch + 1) % 10 == 0:
        torch.save(netG.state_dict(), './%s/netG_epoch_%d.pth' % (opt.save_model_dir, epoch))
        torch.save(netE.state_dict(), './%s/netE_epoch_%d.pth' % (opt.save_model_dir, epoch))
        torch.save(netD.state_dict(), './%s/netD_epoch_%d.pth' % (opt.save_model_dir, epoch))

        dataset_fake = generate_sample(generator = netG, latent_size = latent_size)
        fid = calculate_fid(dataset_fake, m_true, s_true)

        print("The Frechet Inception Distance:", fid)

        vutils.save_image(d_fake.cpu().data[:16, ], './%s/fake_%d.png' % (opt.save_image_dir, epoch))
