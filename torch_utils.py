import torch
import torch.nn as nn
from torch.autograd import Variable
import utils


def assert_zero_grads(params):
    for p in params:
        if p.grad is not None:
            utils.assert_eq(p.grad.data.sum(), 0)


def weights_init(m):
    """custom weights initialization called on net_g and net_f."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def init_net(net, net_file):
    if net_file:
        net.load_state_dict(torch.load(net_file))
    else:
        net.apply(weights_init)
