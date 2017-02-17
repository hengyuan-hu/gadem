from __future__ import print_function
import argparse
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
from torch.autograd import Variable
import os
import numpy as np

import models.dcgan as dcgan
import models.mlp as mlp

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nepoch', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
# parser.add_argument('--clamp_lower', type=float, default=-0.01)
# parser.add_argument('--clamp_upper', type=float, default=0.01)
# parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
opt = parser.parse_args()
print(opt)

if opt.experiment is None:
    opt.experiment = 'samples'
os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3
n_extra_layers = int(opt.n_extra_layers)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

if opt.noBN:
    netG = dcgan.DCGAN_G_nobn(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
elif opt.mlp_G:
    netG = mlp.MLP_G(opt.imageSize, nz, nc, ngf, ngpu)
else:
    netG = dcgan.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)

netG.apply(weights_init)
if opt.netG != '': # load checkpoint if needed
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

if opt.mlp_D:
    netD = mlp.MLP_D(opt.imageSize, nz, nc, ndf, ngpu)
else:
    netD = dcgan.DCGAN_D(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers)
    netD.apply(weights_init)

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netD.cuda()
    netG.cuda()
    input = input.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

input = Variable(input)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)

# setup optimizer
if opt.adam:
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
else:
    optimizerD = optim.RMSprop(netD.parameters(), lr = opt.lrD)
    optimizerG = optim.RMSprop(netG.parameters(), lr = opt.lrG)


loss_d = np.zeros(len(dataloader))
loss_g = np.zeros(len(dataloader))
fe_d_real = np.zeros(len(dataloader))
fe_d_fake = np.zeros(len(dataloader))

for epoch in range(opt.nepoch):
    data_iter = iter(dataloader)
    for b in range(len(dataloader)):
        pcd_k = 5

        # update G network, draw samples
        for p in netD.parameters():
            p.requires_grad = False # to avoid computation

        for _ in range(pcd_k):
            netG.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            noise.data.resize_(opt.batchSize, nz, 1, 1)
            noise.data.normal_(0, 1)
            fake = netG(noise)
            errG = netD(fake)
            errG.backward(one)
            optimizerG.step()
        loss_g[b] = errG.data[0]

        # update D network, constrastive bp
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False in netG update

        data = data_iter.next()
        # train with real
        real_cpu, _ = data
        netD.zero_grad()
        batch_size = real_cpu.size(0)
        input.data.resize_(real_cpu.size()).copy_(real_cpu)

        errD_real = netD(input)
        errD_real.backward(one)

        # train with fake
        noise.data.resize_(batch_size, nz, 1, 1)
        noise.data.normal_(0, 1)
        fake = netG(noise)
        input.data.copy_(fake.data)
        errD_fake = netD(input)
        errD_fake.backward(mone)
        errD = errD_real - errD_fake
        optimizerD.step()

        loss_d[b] = errD.data[0]
        fe_d_real[b] = errD_real.data[0]
        fe_d_fake[b] = errD_fake.data[0]

    print('[%d/%d] Loss_D: %f Loss_G: %f Loss_D_real: %f Loss_D_fake %f'
          % (epoch, opt.nepoch, loss_d.mean(), loss_g.mean(),
             fe_d_real.mean(), fe_d_fake.mean()))
    # save samples
    fake = netG(fixed_noise)
    vutils.save_image(
        fake.data, '{0}/fake_samples_{1}.png'.format(opt.experiment, epoch), nrow=10)

    # do checkpointing
    # torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.experiment, epoch))
    # torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.experiment, epoch))
