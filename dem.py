from __future__ import print_function

# import torch.nn as nn
import os
import time
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.utils as vutils

import utils
import torch_utils


def append_adversarial_x(x_cpu, net_f, eps):
    # TODO: assert net_f gradient should be zero before entering
    utils.assert_eq(type(x_cpu), torch.FloatTensor)
    x_gpu = x_cpu.cuda()
    x = Variable(x_gpu, requires_grad=True)
    fe = net_f(x)
    # print('free energy of original x:', fe.data[0])
    fe.backward()
    scaled_grad_sign = torch.sign(x.grad.data) * eps
    adv_x = x.data + scaled_grad_sign
    adv_x.clamp_(-1.0, 1.0)
    # print('==== after generating adv examples ====')
    # print('free energy of original x:', net_f(x).data[0])
    # print('free energy of adv x:', net_f(Variable(adv_x)).data[0])
    # print('type of x.data:', type(x.data))
    ret = torch.cat((x_gpu, adv_x))
    # print('shape of combined tensor:', ret.size())
    # vutils.save_image(
    #     x.data, 'tests/eps_%s_normal.png' % eps, nrow=10)
    # vutils.save_image(
    #     adv_x, 'tests/eps_%s_adversa.png' % eps, nrow=10)
    return ret


class DEM(object):
    def __init__(self, net_f):
        self.net_f = net_f
        # self.dataset = dataset

    def train(self, configs, dataset, sampler):
        optimizer = torch.optim.RMSprop(self.net_f.parameters(), lr=configs.lr_f)
        loss_vals = np.zeros(len(dataset))
        fe_real_vals = np.zeros(len(dataset))
        fe_fake_vals = np.zeros(len(dataset))
        pcd_steps = np.zeros(len(dataset))
        log_file = open(os.path.join(configs.experiment, 'log_f.txt'), 'w')

        # cuda variables and tensors
        x_node = torch_utils.create_cuda_variable(
            (configs.batch_size,) + dataset.x_shape)
        # fixed_noise = torch.FloatTensor(
        #     configs.batch_size, configs.num_z, 1, 1).normal_(0, 1).cuda()
        # one = torch.FloatTensor([1]).cuda()
        # mone = one * -1

        for eid in range(configs.num_epochs):
            dataloader = iter(dataset)
            t = time.time()
            for bid in range(len(dataloader)):
                self.net_f.zero_grad()

                real_cpu = dataloader.next()
                if configs.use_adversarial_real:
                    real_gpu = append_adversarial_x(real_cpu, self.net_f, configs.eps)
                    self.net_f.zero_grad() # necessary?
                    x_node.data.resize_(real_gpu.size()).copy_(real_gpu)
                else:
                    x_node.data.resize_(real_cpu.size()).copy_(real_cpu)
                fe_real = self.net_f(x_node)

                pcd_k = configs.pcd_k if eid < 25 else 100
                samples, pcd_steps[bid] = sampler.sample(
                    self.net_f, fe_real.data[0], pcd_k)
                utils.assert_eq(type(samples), torch.cuda.FloatTensor)

                x_node.data.resize_(samples.size()).copy_(samples)
                fe_fake = self.net_f(x_node)
                loss_f = fe_real - fe_fake
                loss_f.backward()
                optimizer.step()

                fe_real_vals[bid] = fe_real.data[0]
                fe_fake_vals[bid] = fe_fake.data[0]
                loss_vals[bid] = loss_f.data[0]

            log = '[%d/%d] Loss_F: %f FE_Real: %f FE_Fake %f, PCD-K %f' \
                  % (eid+1, configs.num_epochs, loss_vals.mean(),
                     fe_real_vals.mean(), fe_fake_vals.mean(), pcd_steps.mean())
            print(log)
            print('Time Taken:', time.time() - t)
            log_file.write(log+'\n')
            log_file.flush()
            sampler.save_samples(os.path.join(configs.experiment, 'epoch%d' % (eid+1)))

            if (eid+1) % 10 == 0:
                torch.save(self.net_f.state_dict(),
                           '%s/net_f_epoch_%s.pth' % (configs.experiment, eid+1))
                torch.save(sampler.net_g.state_dict(),
                           '%s/net_g_epoch_%s.pth' % (configs.experiment, eid+1))

        log_file.close()

    def eval(self, train_xs, test_xs):
        batch_size = 1000
        utils.assert_eq(len(train_xs) % batch_size, 0)
        utils.assert_eq(len(test_xs) % batch_size, 0)
        train_fes = np.zeros(len(train_xs) // batch_size)
        test_fes = np.zeros(len(test_xs) // batch_size)

        x_node = torch_utils.create_cuda_variable((batch_size,) + dataset.x_shape)
        for i in range(len(train_fes)):
            x_node.data.copy_(train_xs[i*batch_size : (i+1)*batch_size])
            train_fes[i] = self.net_f(x_node)
        for i in range(len(test_fes)):
            x_node.data.copy_(test_xs[i*batch_size, (i+1)*batch_size])
            test_fes[i] = self.net_f(x_node)

        mean_train_fes = train_fes.mean()
        mean_test_fes = test_fes.mean()
        print('Eval: free_energy on train: %s, free_energy on test: %s, ratio: %s' %
              (mean_train_fes, mean_test_fes, mean_train_fes / mean_test_fes))


class Sampler(object):
    def __init__(self, net_g, num_z, lr, num_samples, x_shape):
        # self.num_z = num_z
        # self.lr = lr
        # self.num_samples = num_samples
        self.net_g = net_g
        self.fix_noise = torch_utils.create_cuda_variable(
            (num_samples, num_z, 1, 1))
        self.fix_noise.data.normal_(0, 1)
        self.noise = torch_utils.create_cuda_variable(
            (num_samples, num_z, 1, 1))
        self.lmc_samples = torch_utils.create_cuda_variable(
            (num_samples,)+x_shape)
        self.lmc_samples.data.normal_(0, 1)
        self.optimizer = torch.optim.RMSprop(self.net_g.parameters(), lr=lr)

    def sample(self, net_f, lower_bound, pcd_k):
        for p in net_f.parameters():
            p.requires_grad = False

        self.noise.data.normal_(0, 1)
        for step in range(pcd_k):
            self.net_g.zero_grad()
            fake = self.net_g(self.noise)
            fe = net_f(fake)
            # print('fake fe:', fe.data[0])
            # print('real fe:', lower_bound)
            if fe.data[0] < lower_bound:
                break
            fe.backward()
            self.optimizer.step()

        for p in net_f.parameters():
            p.requires_grad = True

        # print('type of fake: (guess: Variable)', type(fake.data))
        # TODO: LMC samples
        return fake.data, step

    def save_samples(self, prefix):
        fake = self.net_g(self.fix_noise)
        vutils.save_image(
            fake.data, '%s_fake_samples.png' % prefix, nrow=10)
