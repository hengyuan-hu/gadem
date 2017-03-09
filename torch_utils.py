import torch
# import torch.nn as nn
# import torch.nn.parallel
from torch.autograd import Variable


def create_cuda_variable(shape):
    var = Variable(torch.FloatTensor(*shape))
    return var.cuda()
