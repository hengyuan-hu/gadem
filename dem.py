from __future__ import print_function

import os
import time
import numpy as np
import torch
from torch.autograd import Variable
import torchvision
import utils
import torch_utils


def assert_zero_grads(params):
    for p in params:
        utils.assert_eq(p.grad.data.sum(), 0)


def append_adversarial_x(x_cpu, net_f, eps):
    # assert_zero_grads(net_f.parameters())
    for p in net_f.parameters():
        p.requires_grad = False

    x = Variable(x_cpu.cuda(), requires_grad=True)
    fe = net_f(x)
    fe.backward()
    scaled_grad_sign = torch.sign(x.grad.data) * eps
    adv_x = x.data + scaled_grad_sign
    adv_x.clamp_(-1.0, 1.0)
    ret = torch.cat((x.data, adv_x))
    # print('x energy:', fe.data[0], 'adv_x energy:', net_f(Variable(adv_x)).data[0])

    for p in net_f.parameters():
        p.requires_grad = True
    # assert_zero_grads(net_f.parameters())
    return ret


def lmc_move(samples, net_f, noises, grad_scale, noise_scale):
    utils.assert_eq(type(samples), torch.cuda.FloatTensor)
    utils.assert_eq(type(noises), torch.cuda.FloatTensor)
    for p in net_f.parameters():
        assert not p.requires_grad

    samples = Variable(samples, requires_grad=True)
    fe = net_f(samples)
    fe.backward()

    scaled_grads = grad_scale * samples.grad.data
    scaled_noises = noise_scale * noises
    lmc_samples = samples.data - scaled_grads + scaled_noises
    lmc_samples.clamp_(-1.0, 1.0)
    utils.assert_eq(type(lmc_samples), torch.cuda.FloatTensor)
    return lmc_samples


class DEM(object):
    def __init__(self, net_f):
        self.net_f = net_f

    def train(self, configs, dataset, sampler):
        optimizer = torch.optim.RMSprop(self.net_f.parameters(), lr=configs.lr_f)
        loss_vals = np.zeros(len(dataset))
        fe_pos_vals = np.zeros(len(dataset))
        fe_neg_vals = np.zeros(len(dataset))
        fe_g_vals = np.zeros(len(dataset))
        g_steps = np.zeros(len(dataset))

        # fe_lmc_vals = np.zeros(len(dataset))
        # lmc_steps = np.zeros(len(dataset))

        log_file = open(os.path.join(configs.experiment, 'log_f.txt'), 'w')

        data_node = torch_utils.create_cuda_variable(
            (configs.batch_size,) + dataset.x_shape)
        sample_node = torch_utils.create_cuda_variable(
            (configs.batch_size,) + dataset.x_shape)

        for eid in range(configs.num_epochs):
            dataloader = iter(dataset)
            t = time.time()

            for bid in range(len(dataloader)):
                self.net_f.zero_grad()
                pos_cpu = dataloader.next()

                if configs.use_adv:
                    pos_gpu = append_adversarial_x(
                        pos_cpu, self.net_f, configs.adv_eps)
                    data_node.data.resize_(pos_gpu.size()).copy_(pos_gpu)
                else:
                    data_node.data.resize_(pos_cpu.size()).copy_(pos_cpu)

                fe_pos = self.net_f(data_node)

                # assert_zero_grads(sampler.net_g.parameters())
                pcd_k = configs.pcd_k if eid < 25 else 100
                # use_lmc = eid >= 25
                samples, infos = sampler.sample(
                    self.net_f, fe_pos.data[0], pcd_k, configs.use_lmc)
                g_steps[bid], fe_g_vals[bid] = infos
                utils.assert_eq(type(samples), torch.cuda.FloatTensor)
                # assert_zero_grads(sampler.net_g.parameters())

                sample_node.data.resize_(samples.size()).copy_(samples)
                fe_neg = self.net_f(sample_node)
                loss_f = fe_pos - fe_neg

                # assert_zero_grads(self.net_f.parameters())
                loss_f.backward()
                optimizer.step()

                fe_pos_vals[bid] = fe_pos.data[0]
                fe_neg_vals[bid] = fe_neg.data[0]
                loss_vals[bid] = loss_f.data[0]

            log = ('[%d/%d] Loss_F: %.8f FE_pos: %.8f, FE_neg %.8f, '
                   'FE_g %.8f, g_steps %.1f') \
                   % (eid+1, configs.num_epochs,
                      loss_vals.mean(), fe_pos_vals.mean(), fe_neg_vals.mean(),
                      fe_g_vals.mean(), g_steps.mean())
            print(log)
            print('Time Taken:', time.time() - t)
            log_file.write(log+'\n')
            log_file.flush()
            sampler.save_samples(
                os.path.join(configs.experiment, 'epoch%d' % (eid+1)))

            if (eid+1) % 10 == 0:
                torch.save(self.net_f.state_dict(),
                           '%s/net_f_epoch_%s.pth' % (configs.experiment, eid+1))
                torch.save(sampler.net_g.state_dict(),
                           '%s/net_g_epoch_%s.pth' % (configs.experiment, eid+1))
                eval_log = self.eval(dataset.train_xs, dataset.test_xs)
                log_file.write(eval_log+'\n')
                log_file.flush()

        log_file.close()

    def eval(self, train_xs, test_xs):
        # print(train_xs.shape)
        # print(test_xs.shape)
        batch_size = 1000
        utils.assert_eq(len(train_xs) % batch_size, 0)
        utils.assert_eq(len(test_xs) % batch_size, 0)
        train_fes = np.zeros(len(train_xs) // batch_size)
        test_fes = np.zeros(len(test_xs) // batch_size)

        x_node = torch_utils.create_cuda_variable((batch_size,) + train_xs.shape[1:])
        for i in xrange(len(train_fes)):
            x_node.data.copy_(
                torch.from_numpy(train_xs[i*batch_size : (i+1)*batch_size]))
            train_fes[i] = self.net_f(x_node).data[0]
        for i in xrange(len(test_fes)):
            # print(test_xs.shape)
            x_node.data.copy_(
                torch.from_numpy(test_xs[i*batch_size : (i+1)*batch_size]))
            test_fes[i] = self.net_f(x_node).data[0]

        mean_train_fes = train_fes.mean()
        mean_test_fes = test_fes.mean()
        log = 'Eval:\nfree_energy on train: %s;\nfree_energy on test: %s;\nratio: %s' \
              % (mean_train_fes, mean_test_fes,
                 np.exp(mean_train_fes-mean_test_fes))
        print(log)
        return log


class Sampler(object):
    def __init__(self, net_g, configs, x_shape):
        self.net_g = net_g
        self.optimizer = torch.optim.RMSprop(self.net_g.parameters(), lr=configs.lr_g)

        z_shape = (configs.batch_size, configs.nz, 1, 1)
        self.z = torch_utils.create_cuda_variable(z_shape)
        self.fix_z = torch_utils.create_cuda_variable(z_shape)
        self.fix_z.data.normal_(0, 1)

        samples_shape = (configs.batch_size,) + x_shape
        # self.lmc_samples = torch.FloatTensor(*samples_shape)
        # self.lmc_samples = self.lmc_samples.cuda()
        # self.lmc_samples.uniform_(-1.0, 1.0)

        self.lmc_noise = torch.FloatTensor(*samples_shape).cuda()
        # self.lmc_noise = self.lmc_noise.cuda()
        self.lmc_grad_scale = configs.lmc_grad_scale
        self.lmc_noise_scale = configs.lmc_noise_scale

    def sample(self, net_f, lower_bound, pcd_k, use_lmc):
        for p in net_f.parameters():
            p.requires_grad = False
        assert net_f.training
        net_f.eval()

        self.z.data.normal_(0, 1)
        for g_step in range(pcd_k):
            self.net_g.zero_grad()
            samples = self.net_g(self.z)
            samples_fe = net_f(samples)
            if samples_fe.data[0] < lower_bound:
                break
            samples_fe.backward()
            self.optimizer.step()

        samples = samples.data
        infos = (g_step, samples_fe.data[0])
        if use_lmc:
            self.lmc_noise.normal_(0, 1)
            samples = lmc_move(samples, net_f, self.lmc_noise,
                               self.lmc_grad_scale, self.lmc_noise_scale)

        for p in net_f.parameters():
            p.requires_grad = True
        net_f.train()
        return samples, infos

    def save_samples(self, prefix):
        assert self.net_g.training
        self.net_g.eval()
        samples = self.net_g(self.fix_z)
        self.net_g.train()
        torchvision.utils.save_image(
            samples.data, '%s_g_samples.png' % prefix, nrow=10)
