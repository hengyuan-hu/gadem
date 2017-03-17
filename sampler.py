from __future__ import print_function

import torch
from torch.autograd import Variable
import torchvision
import utils
import torch_utils
import models.dcgan as dcgan


class Sampler(torch.nn.Module):
    def __init__(self, cfgs):
        super(Sampler, self).__init__()

        self.net_g = dcgan.DCGAN_G(
            cfgs.image_size, cfgs.nz, cfgs.nc, cfgs.ngf, cfgs.ngpu)
        self._init_nets(cfgs)

        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=cfgs.lr_g)

        self.avg_fe = 0.0
        self.avg_fe_decay = cfgs.avg_fe_decay

        self.z = Variable(torch.cuda.FloatTensor(cfgs.batch_size, cfgs.nz, 1, 1))
        self.fix_z = Variable(
            torch.cuda.FloatTensor(cfgs.batch_size, cfgs.nz, 1, 1), volatile=True)
        self.fix_z.data.normal_(0, 1)

    def _init_nets(self, cfgs):
        torch_utils.init_net(self.net_g, cfgs.net_g)
        self.net_g.cuda()

    def _update_avg_fe(self, new_fe):
        self.avg_fe = self.avg_fe_decay*self.avg_fe + (1-self.avg_fe_decay)*new_fe

    def sample(self, net_f, lower_bound, max_steps):
        # torch_utils.assert_zero_grads(self.net_g.parameters())
        for p in net_f.parameters():
            p.requires_grad = False
        assert net_f.training
        # net_f.eval()

        self.z.data.normal_(0, 1)
        for g_step in range(max_steps):
            samples = self.net_g(self.z)
            samples_fe = net_f(samples)
            self._update_avg_fe(samples_fe.data[0])
            if samples_fe.data[0] < lower_bound:
            # if self.avg_fe < lower_bound:
                break
            samples_fe.backward()
            self.optimizer.step()
            self.net_g.zero_grad()

        samples = samples.data
        infos = (g_step,)

        for p in net_f.parameters():
            p.requires_grad = True
        # net_f.train()
        # torch_utils.assert_zero_grads(self.net_g.parameters())
        # utils.assert_eq(type(samples), torch.cuda.FloatTensor)
        return samples, infos

    def save_samples(self, prefix):
        assert self.net_g.training
        self.net_g.eval()
        samples = self.net_g(self.fix_z)
        self.net_g.train()
        torchvision.utils.save_image(
            samples.data, '%s_g_samples.png' % prefix, nrow=10)
