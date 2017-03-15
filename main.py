from __future__ import print_function
import os
# import time

import argparse
import random
import torch
# import torch.nn as nn
# import torch.nn.parallel
import torch.backends.cudnn as cudnn
# import torch.optim as optim
# import torch.utils.data
# import torchvision.datasets as dset
# import torchvision.transforms as transforms
# import torchvision.utils as vutils
# from torch.autograd import Variable
# import numpy as np

import models.dcgan as dcgan
# import models.mlp as mlp

from dataset_wrapper import Cifar10Wrapper
from dem import DEM, Sampler


parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', required=True,
#                     help='cifar10 | lsun | imagenet | folder | lfw ')
# parser.add_argument('--dataroot', required=True, help='path to dataset')
# parser.add_argument('--workers', type=int,
#                     help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=100,
                    help='input batch size')
parser.add_argument('--image_size', type=int, default=32,
                    help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='num_channel of images')
parser.add_argument('--nz', type=int, default=100,
                    help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--num_epochs', type=int, default=1000,
                    help='number of epochs to train for')
parser.add_argument('--lr_f', type=float, default=0.0005,
                    help='learning rate for Critic, default=0.00005')
parser.add_argument('--lr_g', type=float, default=0.0005,
                    help='learning rate for Generator, default=0.00005')
# parser.add_argument('--use_lmc', action='store_true', help='use adv examples')
parser.add_argument('--lmc_grad_scale', type=float, default=0.1)
parser.add_argument('--lmc_noise_scale', type=float, default=0.001)
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--net_g', default='', help="path to net_g")
parser.add_argument('--net_f', default='', help="path to net_f")
parser.add_argument('--use_adv', action='store_true', help='use adv examples')
parser.add_argument('--adv_eps', type=float, default=0.01)
parser.add_argument('--experiment', default=None,
                    help='Where to store samples and models')


def weights_init(m):
    """custom weights initialization called on net_g and net_f."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if __name__ == '__main__':
    opt = parser.parse_args()
    opt.manualSeed = 666999
    torch.manual_seed(999666)
    torch.cuda.manual_seed(6669999)
    print(opt)

    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    assert opt.experiment is not None, 'specify output dir to avoid overwriting.'
    if not os.path.exists(opt.experiment):
        os.makedirs(opt.experiment)
    print(opt, file=open(os.path.join(opt.experiment, 'configs.txt'), 'w'))

    cudnn.benchmark = True

    dataset = Cifar10Wrapper.load_default(opt.batch_size)

    # this is tricky, use bn is bad?
    net_f = dcgan.DCGAN_D_nobn(opt.image_size, opt.nz, opt.nc, opt.ndf, opt.ngpu)
    if opt.net_f:
        net_f.load_state_dict(torch.load(opt.net_f))
    else:
        net_f.apply(weights_init)
    print(net_f)

    net_g = dcgan.DCGAN_G_nobn(opt.image_size, opt.nz, opt.nc, opt.ngf, opt.ngpu)
    if opt.net_g:
        net_g.load_state_dict(torch.load(opt.net_g))
    else:
        net_g.apply(weights_init)
    print(net_g)

    net_f.cuda()
    net_g.cuda()

    dem = DEM(net_f)
    sampler = Sampler(net_g, net_f, opt)

    opt.max_g_steps = 100

    if opt.net_f and opt.net_g:
        dem.eval(dataset.train_xs, dataset.test_xs)

    dem.train(opt, dataset, sampler)
