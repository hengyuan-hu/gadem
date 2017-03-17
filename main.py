from __future__ import print_function
import os

import argparse
import torch
import numpy as np
import torch.backends.cudnn as cudnn

from dataset_wrapper import Cifar10Wrapper
from dem import DEM
from sampler import Sampler


parser = argparse.ArgumentParser()
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
parser.add_argument('--lr_f', type=float, default=0.00005,
                    help='learning rate for Critic, default=0.00005')
parser.add_argument('--lr_g', type=float, default=0.00005,
                    help='learning rate for Generator, default=0.00005')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--net_g', default='', help="path to net_g")
parser.add_argument('--net_e', default='', help="path to net_e")
parser.add_argument('--net_f', default='', help="path to net_f")
parser.add_argument('--avg_fe_decay', type=float, default=0.9)
parser.add_argument('--experiment', default=None,
                    help='Where to store samples and models')


if __name__ == '__main__':
    opt = parser.parse_args()
    opt.manualSeed = 666999
    print(opt)

    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    assert opt.experiment is not None, 'specify output dir to avoid overwriting.'
    if not os.path.exists(opt.experiment):
        os.makedirs(opt.experiment)
    print(opt, file=open(os.path.join(opt.experiment, 'configs.txt'), 'w'))

    cudnn.benchmark = True

    dataset = Cifar10Wrapper.load_default(opt.batch_size)
    dem = DEM(opt)
    sampler = Sampler(opt)
    print(dem.net_f)
    print(sampler.net_g)

    opt.max_steps = 25
    dem.train(opt, dataset, sampler)

    # if opt.net_f and opt.net_g:
    #     dem.eval(dataset.train_xs, dataset.test_xs)
