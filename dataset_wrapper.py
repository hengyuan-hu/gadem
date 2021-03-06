"""Wrapper classes for original and encoded datasets."""
from __future__ import print_function

from keras.datasets import mnist, cifar10
import keras.backend as K
import torch

import numpy as np
import matplotlib.pyplot as plt
import h5py
import utils
import math
# import stl_dataset


if K.image_dim_ordering() != 'th':
    print('keras backend: change image ordering to th.')
    K.set_image_dim_ordering('th')


class DatasetWrapper(object):
    idx2cls = []

    def __init__(self, train_xs, train_ys, test_xs, test_ys, batch_size):
        """DO NOT do any normalization in this function"""
        self.train_xs = train_xs.astype(np.float32)
        self.train_ys = train_ys
        self.test_xs = test_xs.astype(np.float32)
        self.test_ys = test_ys
        self.batch_size = batch_size

    def __len__(self):
        return int(math.ceil(1.0 * len(self.train_xs) / self.batch_size))

    def __iter__(self):
        np.random.shuffle(self.train_xs)
        self.batch_idx = 0
        return self

    def next(self):
        self.batch_idx += 1
        batch = self.train_xs[(self.batch_idx-1) * self.batch_size :
                              self.batch_idx * self.batch_size]
        return torch.from_numpy(batch)

    @property
    def x_shape(self):
        return self.train_xs.shape[1:]

    @property
    def cls2idx(self):
        return {cls: idx for (idx, cls) in enumerate(idx2cls)}

    @classmethod
    def load_from_h5(cls, h5_path, batch_size):
        with h5py.File(h5_path, 'r') as hf:
            train_xs = np.array(hf.get('train_xs'))
            train_ys = np.array(hf.get('train_ys'))
            test_xs = np.array(hf.get('test_xs'))
            test_ys = np.array(hf.get('test_ys'))
        print('Dataset loaded from %s' % h5_path)
        return cls(train_xs, train_ys, test_xs, test_ys, batch_size)

    @classmethod
    def load_default(cls, batch_size):
        raise NotImplementedError

    def dump_to_h5(self, h5_path):
        with h5py.File(h5_path, 'w') as hf:
            hf.create_dataset('train_xs', data=self.train_xs)
            hf.create_dataset('train_ys', data=self.train_ys)
            hf.create_dataset('test_xs', data=self.test_xs)
            hf.create_dataset('test_ys', data=self.test_ys)
        print('Dataset written to %s' % h5_path)

    def reshape(self, new_shape):
        batch_size = self.train_xs.shape[0]
        self.train_xs = self.train_xs.reshape((batch_size,) + new_shape)
        batch_size = self.test_xs.shape[0]
        self.test_xs = self.test_xs.reshape((batch_size,) + new_shape)
        assert self.train_xs.shape[1:] == self.test_xs.shape[1:]

    def plot_data_dist(self, fig_path, num_bins=50):
        xs = np.vstack((self.train_xs, self.test_xs))
        if len(xs.shape) > 2:
            num_imgs = len(xs)
            xs = xs.reshape((num_imgs, -1))
        plt.hist(xs, num_bins)
        if fig_path:
            plt.savefig(fig_path)
            plt.close()
        else:
            plt.show()

    # def get_subset(self, subset, subclass=None):
    #     """get a subset.

    #     subset: 'train' or 'test'
    #     subclass: name of the subclass of interest
    #     """
    #     xs = self.train_xs if subset == 'train' else self.test_xs
    #     ys = self.train_ys if subset == 'train' else self.test_ys
    #     assert len(xs) == len(ys)

        # if subclass:
        #     idx = self.cls2idx[subclass]
        #     loc = np.where(ys == idx)[0]
        #     xs = xs[loc]
        #     ys = ys[loc]
        # return xs, ys


class MnistWrapper(DatasetWrapper):
    @classmethod
    def load_default(cls, batch_size):
        ((train_xs, train_ys), (test_xs, test_ys)) = mnist.load_data()
        train_xs = (train_xs / 255.0).reshape(-1, 28, 28, 1)
        test_xs = (test_xs / 255.0).reshape(-1, 28, 28, 1)
        return cls(train_xs, train_ys, test_xs, test_ys, batch_size)


# def augment_batch(batch, pad_dim=4):
#     augmented_batch = np.zeros(batch.shape)
#     num_imgs, _, height, width = batch.shape
#     for i in range(num_imgs):
#         img = batch[i]
#         if np.random.normal(0, 1, (1,))[0] >= 0.5:
#             img = img[:, :, ::-1]
#         img = np.pad(
#             img, ((0,), (pad_dim,), (pad_dim,)), 'constant', constant_values=(0,1))
#         start_h, start_w = np.random.randint(0, 2*pad_dim, (2,))
#         augmented_batch[i] = img[:, start_h:start_h+height, start_w:start_w+width]
#     return augmented_batch


class Cifar10Wrapper(DatasetWrapper):
    idx2cls = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    @classmethod
    def load_default(cls, batch_size):
        ((train_xs, train_ys), (test_xs, test_ys)) = cifar10.load_data()
        # train_xs = utils.preprocess_cifar10(train_xs)
        # test_xs = utils.preprocess_cifar10(test_xs)

        train_xs = (train_xs / 255.0 - 0.5) * 2.0
        test_xs = (test_xs / 255.0 - 0.5) * 2.0
        return cls(train_xs, train_ys, test_xs, test_ys, batch_size)


# class STL10Wrapper(DatasetWrapper):
#     @classmethod
#     def load_default(cls):
#         train_xs = stl_dataset.read_all_images(stl_dataset.UNLABELED_DATA_PATH)
#         train_ys = np.zeros(len(train_xs), dtype=np.uint8)
#         test_xs = stl_dataset.read_all_images(stl_dataset.DATA_PATH)
#         test_ys = stl_dataset.read_labels(stl_dataset.LABEL_PATH)

#         train_xs = utils.preprocess_stl10(train_xs)
#         test_xs = utils.preprocess_stl10(test_xs)
#         return cls(train_xs, train_ys, test_xs, test_ys)


if __name__ == '__main__':
    batch_size = 100
    # mnist_dataset = MnistWrapper.load_default(batch_size)
    # mnist_dataset.plot_data_dist(None)
    cifar10_dataset = Cifar10Wrapper.load_default(batch_size)
    # cifar10_dataset.plot_data_dist(None)
