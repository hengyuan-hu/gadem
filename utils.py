# python utils
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import keras
import keras.backend as K
import math
import os


def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %d (expected)' % (real, expected)


CIFAR10_COLOR_MEAN_RGB = np.array([125.3, 123.0, 113.9]).reshape(3, 1, 1)
CIFAR10_COLOR_STD_RGB  = np.array([63.0,  62.1,  66.7]).reshape(3, 1, 1)


STL10_COLOR_MEAN_RGB = np.array([112.4, 109.0, 98.4]).reshape(3, 1, 1)
STL10_COLOR_STD_RGB = np.array([ 68.5,  66.6,  68.5]).reshape(3, 1, 1)


def preprocess_cifar10(dataset):
    dataset = (dataset - CIFAR10_COLOR_MEAN_RGB) / CIFAR10_COLOR_STD_RGB
    return dataset


def preprocess_stl10(dataset):
    dataset = (dataset - STL10_COLOR_MEAN_RGB) / STL10_COLOR_STD_RGB
    return dataset


def vis_cifar10(imgs, rows, cols, output_name):
    imgs = imgs * CIFAR10_COLOR_STD_RGB + CIFAR10_COLOR_MEAN_RGB
    imgs = np.maximum(np.zeros(imgs.shape), imgs)
    imgs = np.minimum(np.ones(imgs.shape)*255, imgs)
    # print imgs.shape
    imgs = imgs.astype(np.uint8)

    assert imgs.shape[0] == rows * cols
    f, axarr = plt.subplots(rows, cols, figsize=(32, 32))
    for r in range(rows):
        for c in range(cols):
            img = imgs[r * cols + c]
            axarr[r][c].imshow(img)
            axarr[r][c].set_axis_off()
    f.subplots_adjust(hspace=0.2, wspace=0.2)
    if output_name is None:
        plt.show()
    else:
        plt.savefig(output_name)
    plt.close()


def vis_stl10(imgs, rows, cols, output_name):
    imgs = imgs * STL10_COLOR_STD_RGB + STL10_COLOR_MEAN_RGB
    imgs = np.maximum(np.zeros(imgs.shape), imgs)
    imgs = np.minimum(np.ones(imgs.shape)*255, imgs)
    # print imgs.shape
    imgs = imgs.astype(np.uint8)

    assert imgs.shape[0] == rows * cols
    f, axarr = plt.subplots(rows, cols, figsize=(32, 32))
    for r in range(rows):
        for c in range(cols):
            img = imgs[r * cols + c]
            axarr[r][c].imshow(img)
            axarr[r][c].set_axis_off()
    f.subplots_adjust(hspace=0.2, wspace=0.2)
    if output_name is None:
        plt.show()
    else:
        plt.savefig(output_name)
    plt.close()


def vis_mnist(imgs, rows, cols, output_name):
    # TODO: refactor vis_mnist and vis_cifar10 to remove repeated code
    # imgs = np.maximum(np.zeros(imgs.shape), imgs)
    # imgs = np.minimum(np.ones(imgs.shape), imgs)

    # imgs -= imgs.min()
    # print imgs.max()
    # imgs /= imgs.max()

    imgs = imgs.reshape(-1, 28, 28)
    # print '>>> in vis_mnist(), min: %.4f, max: %.4f' % (imgs.min(), imgs.max())
    # assert imgs.min() >= 0 and imgs.max() <= 1
    assert imgs.shape[0] == rows * cols, \
        'num images does not match %d vs %d' % (imgs.shape[0], rows * cols)

    f, axarr = plt.subplots(rows, cols, figsize=(28, 28))
    for r in range(rows):
        for c in range(cols):
            img = imgs[r * cols + c]
            axarr[r][c].imshow(img, cmap='Greys_r')
            axarr[r][c].set_axis_off()
    f.subplots_adjust(hspace=0.2, wspace=0.2)
    if output_name:
        plt.savefig(output_name)
    else:
        plt.show()
    plt.close()


def vis_samples(imgs, rows, cols, img_shape, output_name):
    return vis_weights(imgs.T, rows, cols, img_shape, output_name, 'Greys_r')


def vis_weights(weights, rows, cols, neuron_shape, output_name=None, cmap='Greys'):
    assert weights.shape[-1] == rows * cols
    f, axarr = plt.subplots(rows, cols, figsize=neuron_shape)
    for r in range(rows):
        for c in range(cols):
            neuron_idx = r * cols + c
            weight_map = weights[:, neuron_idx].reshape(neuron_shape)
            axarr[r][c].imshow(weight_map, cmap=cmap)
            axarr[r][c].set_axis_off()
    f.subplots_adjust(hspace=0.2, wspace=0.2)
    if output_name is None:
        plt.show()
    else:
        plt.savefig(output_name)
    plt.close()


def factorize_number(n):
    i = int(math.floor(math.sqrt(n)))
    for k in range(i, 0, -1):
        if n % k == 0:
            j = n / k
            break
    return j, i
