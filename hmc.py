from __future__ import print_function

import os
import time
import numpy as np
import torch
from torch.autograd import Variable

import utils
import torch_utils
import dem
import time


def kinetic(vel):
    batch_size = vel.size()[0]
    vel_sq = torch.mul(vel, vel)
    return 0.5 * vel_sq.view(batch_size, -1).sum(1)


def hamil(pos, vel, potential_fn):
    # utils.assert_eq(type(pos), torch.cuda.FloatTensor)
    # utils.assert_eq(type(vel), torch.cuda.FloatTensor)
    potential = potential_fn(Variable(pos)).data
    return potential + kinetic(vel)


def mh_accept(prev_energy, next_energy):
    """should take tensor as input."""
    uniform = torch.cuda.FloatTensor(*prev_energy.size()).uniform_()
    energy_diff = prev_energy - next_energy
    accept = energy_diff.exp_() >= uniform
    return accept


def simulate_dynamics(init_pos, init_vel, step_size, num_steps, potential_fn):
    """potential fn takes a Variable and returns a Variable."""
    var_pos = Variable(init_pos, requires_grad=True)
    potential = potential_fn(var_pos).sum(0)
    potential.backward()
    new_vel = init_vel - 0.5 * step_size * var_pos.grad.data
    new_pos = init_pos + step_size * new_vel
    new_pos.clamp_(-dem.NORMAL_THRES, dem.NORMAL_THRES)

    for _ in xrange(num_steps - 1):
        var_pos = Variable(new_pos, requires_grad=True)
        potential = potential_fn(var_pos).sum(0)
        potential.backward()
        new_vel.add_(-step_size * var_pos.grad.data)
        new_pos.add_(step_size * new_vel)

        new_pos.clamp_(-dem.NORMAL_THRES, dem.NORMAL_THRES)

    var_pos = Variable(new_pos, requires_grad=True)
    potential = potential_fn(var_pos).sum(0)
    potential.backward()
    new_vel = new_vel - 0.5 * step_size * var_pos.grad.data
    return new_pos, new_vel


def hmc_sample(pos, step_size, num_steps, potential_fn):
    vel = torch.cuda.FloatTensor(*pos.size()).normal_()
    final_pos, final_vel = simulate_dynamics(
        pos, vel, step_size, num_steps, potential_fn)
    accept = mh_accept(
        hamil(pos, vel, potential_fn),
        hamil(final_pos, final_vel, potential_fn))

    accept = accept.float()
    accept_rate = accept.mean()
    accept = accept.view(-1, 1, 1, 1).expand_as(final_pos)
    new_pos = final_pos * accept + pos * (1 - accept)
    return new_pos, accept_rate


class HMCSampler(object):
    def __init__(self, pos,
                 potential_fn,
                 step_size=0.01,
                 num_steps=20,
                 target_accept_rate=0.9,
                 step_min=0.0001,
                 step_max=1.0,
                 step_dec=0.98,
                 step_inc=1.02,
                 avg_accept_slowness=0.9):
        self.pos = pos
        self.potential_fn = potential_fn
        self.step_size = step_size
        self.target_accept_rate = target_accept_rate
        self.num_steps = 20
        self.step_min = step_min
        self.step_max = step_max
        self.step_dec = step_dec
        self.step_inc = step_inc
        self.avg_accept_slowness=0.9
        self.avg_accept_rate = target_accept_rate

    def sample(self, normalize=False):
        self.pos, accept_rate = hmc_sample(
            self.pos, self.step_size, self.num_steps, self.potential_fn)
        self.avg_accept_rate = (self.avg_accept_rate * self.avg_accept_slowness
                                + accept_rate * (1 - self.avg_accept_slowness))
        if self.avg_accept_rate > self.target_accept_rate:
            self.step_size *= self.step_inc
            self.step_size = min(self.step_size, self.step_max)
        else:
            self.step_size *= self.step_dec
            self.step_size = max(self.step_size, self.step_min)

        # if normalize:
        #     self.pos = (self.pos - self.pos.mean()) / self.pos.std()

        return self.pos, accept_rate


def sampler_on_nd_gaussian(burnin, num_chains, num_samples, dim):
    mu = np.random.uniform(0, 20, dim).astype(np.float32)
    cov = np.random.uniform(0, 10, (dim, dim)).astype(np.float32)
    cov = np.dot(cov, cov.T)
    cov = cov / cov.max()
    # cov = np.identity(dim).astype(np.float32)
    cov = (cov + cov.T) / 2.
    cov[np.arange(dim), np.arange(dim)] = 1.0
    cov_inv = np.linalg.inv(cov)


    cuda_mu = torch.from_numpy(mu).cuda()
    cuda_mu = Variable(cuda_mu.view(1, -1).expand(num_chains, cuda_mu.size()[-1]),
                       requires_grad=False)
    cuda_cov_inv = Variable(torch.from_numpy(cov_inv).cuda(),
                            requires_grad=False)

    def gaussian_energy(x):
        e = torch.sum(
            torch.mul(
                torch.mm(x-cuda_mu, cuda_cov_inv),
                x-cuda_mu),
            1)
        # print(e.mean(0).size())
        return 0.5 * e # .mean(0)

    new_pos = torch.cuda.FloatTensor(num_chains, dim).normal_()
    step_size = 0.01
    num_steps = 20

    hmc_sampler = HMCSampler(new_pos, gaussian_energy, step_size, num_steps)

    for _ in xrange(burnin):
        hmc_sampler.sample()
        # new_pos, _, = hmc_sample(new_pos, step_size, num_steps, gaussian_energy)

    samples = []
    accept_rate = []
    for _ in xrange(num_samples):
        # new_pos, rate = hmc_sample(new_pos, step_size, num_steps, gaussian_energy)
        new_pos, rate = hmc_sampler.sample()
        # print(type(new_pos))
        samples.append(new_pos.cpu().numpy())
        accept_rate.append(rate)

    samples = np.array(samples)
    print(samples.shape)
    samples = samples.T.reshape(dim, -1).T
    print(samples.shape)

    print('****** TARGET VALUES ******')
    print('target mean:', mu)
    print('target cov:\n', cov)

    print('****** EMPIRICAL MEAN/COV USING HMC ******')
    print('empirical mean: ', samples.mean(0))
    print('empirical_cov:\n', np.cov(samples.T))

    print('****** HMC INTERNALS ******')
    # print('final stepsize:', final_stepsize
    print('avg acceptance_rate', np.array(accept_rate).mean())
    print('min acceptance_rate', np.array(accept_rate).min())
    print('max acceptance_rate', np.array(accept_rate).max())

    print('DIFF')
    print(np.abs(cov - np.cov(samples.T)).sum())
    # print(cov.sum() - np.cov(samples.T).sum())


if __name__ == '__main__':
    torch.backends.cudnn.benckmark = True
    np.random.seed(666)
    torch.cuda.manual_seed(666999)

    sampler_on_nd_gaussian(1000, 100, 100, 5)
    print('>>>', t_sum)
