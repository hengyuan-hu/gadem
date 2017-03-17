from __future__ import print_function

import os
import time
import numpy as np
import torch
from torch.autograd import Variable
import utils
import torch_utils
import models.dcgan as dcgan


def ceil_lb(val, rate):
    assert rate <= 1 and rate >= 0
    ceil = val * rate if val < 0 else val / rate
    assert ceil > val
    return ceil


ONE = torch.cuda.FloatTensor([1])
NEG_ONE = ONE * -1


class DEM(object):
    def __init__(self, cfgs):
        self.net_f = dcgan.DCGAN_D(
            cfgs.image_size, cfgs.nc, cfgs.ndf, cfgs.ngpu)
        torch_utils.init_net(self.net_f, cfgs.net_f)
        self.net_f.cuda()

    def train(self, cfgs, dataset, sampler):
        optimizer = torch.optim.RMSprop(self.net_f.parameters(), lr=cfgs.lr_f)
        data_node = Variable(torch.cuda.FloatTensor())
        sample_node = Variable(torch.cuda.FloatTensor())

        losses = np.zeros(len(dataset))
        pos_fes = np.zeros(len(dataset))
        neg_fes = np.zeros(len(dataset))
        g_steps = np.zeros(len(dataset))
        log_file = open(os.path.join(cfgs.experiment, 'log_f.txt'), 'w')

        pos_avg_fe = 0.0
        for eid in range(cfgs.num_epochs):
            dataloader = iter(dataset)
            t = time.time()

            for bid in range(len(dataloader)):
                torch_utils.assert_zero_grads(self.net_f.parameters())

                pos_cpu = dataloader.next()
                data_node.data.resize_(pos_cpu.size()).copy_(pos_cpu)
                pos_fe = self.net_f(data_node)
                pos_fe.backward(ONE)
                pos_avg_fe = (pos_avg_fe * cfgs.avg_fe_decay
                              + pos_fe.data[0] * (1 - cfgs.avg_fe_decay))

                max_steps = cfgs.max_steps if eid < 25 else 100
                samples, infos = sampler.sample(
                    self.net_f, pos_fe.data[0], max_steps)
                sample_node.data.resize_(samples.size()).copy_(samples)
                neg_fe = self.net_f(sample_node)
                neg_fe.backward(NEG_ONE)

                optimizer.step()
                self.net_f.zero_grad()

                pos_fes[bid] = pos_fe.data[0]
                neg_fes[bid] = neg_fe.data[0]
                losses[bid] = pos_fe.data[0] - neg_fe.data[0]
                g_steps[bid] = infos[0]

            log = ('[%d/%d] Loss_F: %.8f pos_fe: %.8f, neg_fe: %.8f, g_steps: %.1f\n'
                   % (eid+1, cfgs.num_epochs, losses.mean(),
                      pos_fes.mean(), neg_fes.mean(), g_steps.mean()))
            log += ('\tpos_avg_fe: %.8f, neg_avg_fe: %.8f'
                    % (pos_avg_fe, sampler.avg_fe))
            print(log)
            print('\tTime Taken:', time.time() - t)
            log_file.write(log+'\n')
            log_file.flush()
            sampler.save_samples(
                os.path.join(cfgs.experiment, 'epoch%d' % (eid+1)))

            if (eid+1) % 10 == 0:
                eval_log = self.eval(dataset.train_xs, dataset.test_xs)
                log_file.write(eval_log+'\n')
                log_file.flush()
            if (eid+1) % 50 == 0:
                torch.save(self.net_f.state_dict(),
                           '%s/net_f_epoch_%s.pth' % (cfgs.experiment, eid+1))
                torch.save(sampler.net_g.state_dict(),
                           '%s/net_g_epoch_%s.pth' % (cfgs.experiment, eid+1))

        log_file.close()

    def eval(self, train_xs, test_xs):
        # print(train_xs.shape)
        # print(test_xs.shape)
        batch_size = 1000
        utils.assert_eq(len(train_xs) % batch_size, 0)
        utils.assert_eq(len(test_xs) % batch_size, 0)
        train_fes = np.zeros(len(train_xs) // batch_size)
        test_fes = np.zeros(len(test_xs) // batch_size)

        x_shape = (batch_size,) + train_xs.shape[1:]
        x_node = Variable(torch.cuda.FloatTensor(*x_shape))

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
        log = 'Eval:\n'
        log += '\tfree_energy on train: %s;\n' % mean_train_fes
        log += '\tfree_energy on test: %s;\n' % mean_test_fes
        log += '\tratio: %s' % np.exp(mean_train_fes-mean_test_fes)
        print(log)
        return log
